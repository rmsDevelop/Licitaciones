#Sirve a demanda: recibe un conjunto de licitaciones (esquema raw de scrape.py:
#41 raw + INT-* de calidad; extracto de licitaciones.parquet o salida del
#scraper) y devuelve esas mismas filas con las celdas de inferencia rellenas
#(5 preds + version). Quien llama decide que filas manda (api.py, el flujo
#operativo o el CLI); este archivo no toca ml_estado (eso es de
#Modeling/cleaning.py) ni escribe en Data/licitaciones.parquet.
#
#Receta por conjunto (licitaciones | menores), con los boosters + metas de
#Models/ (contrato de servido del BUILDLOG):
#   1. featurizar con las funciones de Modeling/featurer.py — mismo codigo que
#      en entrenamiento, sin copia. row_missing_count se recalcula contando
#      solo raw+calidad: los objetivos, presentes en entrenamiento, serian NaN
#      sistematicos (+2/3) en todo lo servido;
#   2. hist_volume_*: recuentos prior-ano de la tabla train (features.parquet)
#      reindexados a las filas recibidas — replica exacta de lo que vio el
#      entrenamiento (segmento nuevo -> NaN);
#   3. matriz por linea segun el meta: features en orden, TE por mapas (nivel
#      nuevo -> gmean), categoricas con niveles congelados (nuevo -> missing),
#      resto float32;
#   4. num -> expm1 + clip [0,50]; zd -> prob; disc -> clip [0,70];
#   5. router: zero_discount_pred = prob >= UMBRAL_ZERO_DISCOUNT[conjunto];
#      system = 0 donde zd_pred, si no discount. version = max(meta['creado'])
#      de las 6 lineas: identifica el modelo que predijo.

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling"))
import featurer

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
STEMS = {  # conjunto -> (num_ofertas, zero_discount, discount) en Models/
    "licitaciones": ("licitaciones_num_ofertas", "licitaciones_zero_discount",
                     "licitaciones_discount"),
    "menores": ("menores_num_ofertas", "menores_zero_discount", "menores_discount"),
}

# Umbral del router zero_discount -> 0 por conjunto: el que minimiza el MAE del
# sistema con gate sobre la ventana VAL de features.parquet, derivado el
# 2026-09-02 con los boosters de produccion (curva plana alrededor del optimo):
#   licitaciones 0.475: MAE 6.14 -> 5.96 (precision_zero 0.90, recall_zero 0.93)
#   menores      0.585: MAE 1.18 -> 1.15 (precision_zero 0.98, recall_zero 0.98)
UMBRAL_ZERO_DISCOUNT = {"licitaciones": 0.475, "menores": 0.585}

PREDS = ["num_ofertas_pred", "zero_discount_prob", "zero_discount_pred",
         "discount_pct_pred", "system_discount_pct_pred"]

# Segmentos de los hist_volume_* por conjunto (los de FEATURES_LINEA).
SEGS_HIST = {
    "licitaciones": [["nuts3"], ["cpv_division"], ["procedimiento_code"]],
    "menores": [["nuts3"], ["cpv_division"], ["tipo_contrato_code"]],
}


# ---------------------------------------------------------------------------
# Herramientas de servido
# ---------------------------------------------------------------------------
def version_modelos(models_dir: Path) -> str:
    """max(meta['creado']) de las 6 lineas: el set se entrena en una tanda."""
    metas = (json.loads((models_dir / f"{stem}.meta.json").read_text())
             for stems in STEMS.values() for stem in stems)
    return max(m["creado"] for m in metas)


def cargar_hist_base(feats_path: Path, conjunto: str) -> dict[str, pd.Series]:
    """Serie prior-ano por segmento (indice segs+ano de la tabla train).

    Replica add_prior_year_volumes de featurer sobre features.parquet: para
    cada combinacion (segmento, ano), cuantas filas train del segmento tienen
    ano menor.
    """
    segs = SEGS_HIST[conjunto]
    unicos = sorted({s for seg in segs for s in seg})
    base = pd.read_parquet(feats_path, columns=["ano"] + unicos,
                           filters=[("conjunto", "=", conjunto)])
    for c in unicos:
        base[c] = base[c].astype("string")
    out = {}
    for seg in segs:
        g = base.groupby(seg + ["ano"], observed=True).size()
        out["hist_volume_" + seg[0]] = g.groupby(level=seg).cumsum() - g
    return out


def aplicar_hist(fe: pd.DataFrame, hist: dict[str, pd.Series]) -> pd.DataFrame:
    """Sobrescribe los hist_volume_* con los recuentos de la tabla train.

    featurizar_conjunto los calculo dentro del conjunto recibido (escala
    equivoca en un serve pequeño); aqui se sustituyen por la base train.
    """
    fe = fe.copy()
    for name, prior in hist.items():
        seg = list(prior.index.names)
        keys = fe[seg].copy()
        for c in seg[:-1]:
            keys[c] = keys[c].astype("string")
        idx = pd.MultiIndex.from_frame(keys)
        fe[name] = pd.to_numeric(prior.reindex(idx).to_numpy(), errors="coerce")
        # En entrenamiento el recuento se calculaba pre-cast: segmento NaN ->
        # fuera del groupby -> hist NaN. Post-cast ese NaN es "missing" y el
        # reindex rescataria el recuento de un bucket que no existia en train;
        # se enmascara para replicar exactamente.
        na_seg = (fe[seg].astype("string") == "missing").any(axis=1).to_numpy()
        fe.loc[na_seg, name] = np.nan
    return fe


def matriz_linea(fe: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """X segun el contrato del meta (features en orden, TE por mapas,
    categoricas con niveles congelados, resto float32)."""
    te = meta.get("target_encoding")
    X = pd.DataFrame(index=fe.index)
    for c in meta["features"]:
        if te and c.endswith("_te"):
            raw = c[:-3]
            X[c] = (fe[raw].astype("string").map(te["maps"][raw])
                    .fillna(te["gmean"]).astype("float32"))
        else:
            X[c] = fe[c]
    for c, niveles in meta["categorical_levels"].items():
        s = X[c].astype("string")
        X[c] = pd.Categorical(s.where(s.isin(niveles)), categories=niveles)
    for c in X.columns:
        if str(X[c].dtype) != "category":
            X[c] = X[c].astype("float32")
    return X


def predecir_linea(fe: pd.DataFrame, models_dir: Path, stem: str) -> np.ndarray:
    """Prediccion final (transform + clip del meta) de una linea."""
    meta = json.loads((models_dir / f"{stem}.meta.json").read_text())
    te = meta.get("target_encoding")
    faltan = [c if not (te and c.endswith("_te")) else f"{c} (desde {c[:-3]})"
              for c in meta["features"]
              if (c[:-3] if (te and c.endswith("_te")) else c) not in fe.columns]
    if faltan:
        raise ValueError(f"{meta['linea']}: faltan features {faltan} — "
                         "¿la entrada salio de scrape.py?")

    bst = xgb.Booster()
    bst.load_model(str(models_dir / f"{stem}.ubj"))
    pred = bst.predict(xgb.DMatrix(matriz_linea(fe, meta), enable_categorical=True))
    del bst
    gc.collect()

    pred = np.asarray(pred, dtype=np.float64)
    if meta["transform"] == "log1p":
        pred = np.expm1(pred)
    lo, hi = meta["clip"] or (-np.inf, np.inf)
    return np.clip(pred, lo, hi)


# ---------------------------------------------------------------------------
# Servido de un conjunto de filas
# ---------------------------------------------------------------------------
def featurizar_servido(sub: pd.DataFrame, conjunto: str,
                       feats_path: Path) -> pd.DataFrame:
    """Filas raw de un conjunto -> tabla de features de servido (pasos 1-2)."""
    sub = sub.reset_index(drop=True)
    sub["ano"] = featurer.fix_ano(sub["ano"]).round().astype("Int64")
    try:
        fe = featurer.featurizar_conjunto(sub, conjunto)
    except KeyError as e:
        raise ValueError(f"falta la columna raw {e} — la entrada debe salir "
                         "de scrape.py") from e

    # row_missing_count contando solo las columnas de entrada (sin objetivos):
    # en entrenamiento los objetivos, presentes, aportaban 0; a NaN en servido
    # sumarian +2/3 sistematicos en toda fila.
    conteo = [c for c in sub.columns
              if c not in featurer.OBJETIVOS + ["ml_estado"] + PREDS + ["version"]]
    rmc = sub[conteo].isna().sum(axis=1).astype("float64")
    fe["row_missing_count"] = rmc.to_numpy()
    return aplicar_hist(fe, cargar_hist_base(feats_path, conjunto))


def inferir(df: pd.DataFrame, models_dir: Path = Path("Models"),
            feats_path: Path = Path("Data/features.parquet")) -> pd.DataFrame:
    """Filas raw (+calidad, columna 'conjunto') -> mismas filas + PREDS + version.

    Se sirve toda fila recibida (sin filtro de poblacion: en servido no hay
    objetivo que validar). La salida queda agrupada por conjunto. ml_estado,
    si viene, pasa intacto.
    """
    t0 = time.time()
    if "conjunto" not in df.columns:
        raise ValueError("la entrada necesita columna 'conjunto'")
    desconocidos = set(df["conjunto"].dropna().unique()) - set(STEMS)
    if desconocidos:
        raise ValueError(f"conjuntos desconocidos: {sorted(desconocidos)}")
    if not feats_path.exists():
        raise ValueError(f"no existe {feats_path}: es la base de los hist_volume "
                         "(regenerar con Modeling/featurer.py)")

    df = df.drop(columns=[c for c in PREDS + ["version"] if c in df.columns])
    version = version_modelos(models_dir)

    partes = []
    for conjunto, sub in df.groupby("conjunto", observed=True):
        sub = sub.reset_index(drop=True)
        fe = featurizar_servido(sub, conjunto, feats_path)

        stem_num, stem_zd, stem_disc = STEMS[conjunto]
        num = predecir_linea(fe, models_dir, stem_num)
        prob = predecir_linea(fe, models_dir, stem_zd)
        disc = predecir_linea(fe, models_dir, stem_disc)

        zd = prob >= UMBRAL_ZERO_DISCOUNT[conjunto]
        sub["num_ofertas_pred"] = num.astype("float32")
        sub["zero_discount_prob"] = prob.astype("float32")
        sub["zero_discount_pred"] = zd.astype("int8")
        sub["discount_pct_pred"] = disc.astype("float32")
        sub["system_discount_pct_pred"] = np.where(zd, 0.0, disc).astype("float32")
        sub["version"] = version
        print(f"  {conjunto}: {len(sub):,} filas servidas | "
              f"zd_pred {zd.mean():.1%} (umbral {UMBRAL_ZERO_DISCOUNT[conjunto]})")
        partes.append(sub)
        del fe
        gc.collect()

    res = pd.concat(partes, ignore_index=True)
    print(f"version {version} | {time.time() - t0:.0f}s")
    return res


def main() -> None:
    p = argparse.ArgumentParser(
        description="Servir un conjunto de licitaciones (raw) con los boosters de Models/.")
    p.add_argument("--input", required=True, help="parquet raw (+INT) con columna 'conjunto'")
    p.add_argument("--salida", default=None, help="parquet de salida (default: <input>_preds.parquet)")
    p.add_argument("--models-dir", default="Models")
    p.add_argument("--feats", default="Data/features.parquet",
                   help="tabla train: base de los hist_volume")
    args = p.parse_args()

    entrada = Path(args.input)
    if entrada.suffix != ".parquet":
        raise SystemExit("la entrada debe ser un parquet")
    salida = Path(args.salida) if args.salida else entrada.with_name(entrada.stem + "_preds.parquet")

    res = inferir(pd.read_parquet(entrada), Path(args.models_dir), Path(args.feats))
    res.to_parquet(salida, index=False, compression="snappy")
    print(f"escrito: {salida}  ({len(res):,} filas x {res.shape[1]} columnas)")


if __name__ == "__main__":
    main()
