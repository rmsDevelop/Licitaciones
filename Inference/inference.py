#Sirve a demanda: recibe un conjunto de licitaciones (esquema raw de scrape.py:
#41 raw + INT-* de calidad; extracto de licitaciones.parquet o salida del
#scraper) y devuelve esas mismas filas con las celdas de inferencia rellenas
#(5 preds + version). Quien llama decide que filas manda (api.py, el flujo
#operativo o el CLI); este archivo no toca ml_estado (eso es de
#Modeling/cleaning.py) ni escribe en Data/licitaciones.parquet.
#
#Receta por conjunto (licitaciones | menores), con los boosters + metas de
#Models/ (contrato de servido del BUILDLOG):
#   1. featurizar con Modeling/featurer.py (la capa f_* del Lab) — mismo
#      codigo que en entrenamiento, sin copia y sin dependencia de Data/:
#      las features no aprenden de la tabla train (los hist_volume_* murieron
#      con el Lab viejo);
#   2. matriz por linea segun el meta: features en orden, encoding del organo
#      por mapas (TE: nivel nuevo/nulo -> prior; frecuencia: -> NaN),
#      categoricas con niveles congelados por orden de aparicion (nuevo ->
#      missing), resto float32;
#   3. num -> clip [0,50]; zd -> prob; disc -> clip [0,70] (sin transform:
#      los campeones del Lab regresan el objetivo directo);
#   4. router: zero_discount_pred = prob >= UMBRAL_ZERO_DISCOUNT[conjunto];
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

# Umbral del router zero_discount -> 0 por conjunto: el que minimiza el MAE
# del sistema con gate sobre la ventana VAL, derivado con los boosters de
# produccion (curva plana alrededor del optimo). Valores vigentes derivados
# el 2026-09-05 con los campeones de Nueva_Licitaciones_Lab:
#   licitaciones 0.365: MAE 7.09 -> 7.07 (precision_zero 0.87, recall_zero 0.90)
#   menores      0.555: MAE 1.26 -> 1.24 (precision_zero 0.97, recall_zero 0.98)
UMBRAL_ZERO_DISCOUNT = {"licitaciones": 0.365, "menores": 0.555}

PREDS = ["num_ofertas_pred", "zero_discount_prob", "zero_discount_pred",
         "discount_pct_pred", "system_discount_pct_pred"]


# ---------------------------------------------------------------------------
# Herramientas de servido
# ---------------------------------------------------------------------------
def version_modelos(models_dir: Path) -> str:
    """max(meta['creado']) de las 6 lineas: el set se entrena en una tanda."""
    metas = (json.loads((models_dir / f"{stem}.meta.json").read_text())
             for stems in STEMS.values() for stem in stems)
    return max(m["creado"] for m in metas)


def matriz_linea(fe: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """X segun el contrato del meta: features en orden, organo por su encoding
    (mapas del meta), categoricas con niveles congelados, resto float32."""
    enc = meta.get("encoding_organo") or {}
    # columnas crudas en el orden del meta (la sintetica del organo no esta en
    # fe; se anade despues EN SU SITIO: te/frecuencia sobrescriben la cruda,
    # f_organo_te va al final, su posicion en la lista del campeon).
    X = fe[[c for c in meta["features"] if c in fe.columns]].copy()
    tipo = enc.get("tipo")
    if tipo in ("te", "frecuencia"):
        s = fe[enc["columna"]].astype("string")
        if tipo == "te":
            X[enc["columna"]] = (s.map(enc["maps"]).fillna(enc["prior"])
                                 .astype("float32"))
        else:  # frecuencia: no vistos/nulos -> NaN, sin fill
            X[enc["columna"]] = s.map(enc["maps"]).astype("float32")
    elif tipo == "te_extra":
        s = fe[enc["columna"]].astype("string")
        X["f_organo_te"] = (s.map(enc["maps"]).fillna(enc["prior"])
                            .astype("float32"))

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
    enc = meta.get("encoding_organo") or {}
    tipo = enc.get("tipo")
    # columna que sintetiza matriz_linea desde la cruda del organo (si la hay)
    sintetica = ("f_organo_te" if tipo == "te_extra"
                 else enc.get("columna") if tipo in ("te", "frecuencia") else None)
    faltan = [c for c in meta["features"]
              if c not in fe.columns and c != sintetica]
    if faltan or (sintetica and enc["columna"] not in fe.columns):
        raise ValueError(f"{meta['linea']}: faltan features {faltan or enc['columna']} — "
                         "¿la entrada salio de scrape.py?")

    bst = xgb.Booster()
    bst.load_model(str(models_dir / f"{stem}.ubj"))
    pred = bst.predict(xgb.DMatrix(matriz_linea(fe, meta), enable_categorical=True))
    del bst
    gc.collect()

    pred = np.asarray(pred, dtype=np.float64)
    if meta["transform"] == "log1p":     # ningun campeon del Lab lo usa
        pred = np.expm1(pred)
    lo, hi = meta["clip"] or (-np.inf, np.inf)
    return np.clip(pred, lo, hi)


# ---------------------------------------------------------------------------
# Servido de un conjunto de filas
# ---------------------------------------------------------------------------
def featurizar_servido(sub: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Filas raw de un conjunto -> tabla de features de servido."""
    sub = sub.reset_index(drop=True)
    try:
        return featurer.featurizar_conjunto(sub, conjunto)
    except KeyError as e:
        raise ValueError(f"falta la columna raw {e} — la entrada debe salir "
                         "de scrape.py") from e


def inferir(df: pd.DataFrame, models_dir: Path = Path("Models")) -> pd.DataFrame:
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

    df = df.drop(columns=[c for c in PREDS + ["version"] if c in df.columns])
    version = version_modelos(models_dir)

    partes = []
    for conjunto, sub in df.groupby("conjunto", observed=True):
        sub = sub.reset_index(drop=True)
        fe = featurizar_servido(sub, conjunto)

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
    args = p.parse_args()

    entrada = Path(args.input)
    if entrada.suffix != ".parquet":
        raise SystemExit("la entrada debe ser un parquet")
    salida = Path(args.salida) if args.salida else entrada.with_name(entrada.stem + "_preds.parquet")

    res = inferir(pd.read_parquet(entrada), Path(args.models_dir))
    res.to_parquet(salida, index=False, compression="snappy")
    print(f"escrito: {salida}  ({len(res):,} filas x {res.shape[1]} columnas)")


if __name__ == "__main__":
    main()
