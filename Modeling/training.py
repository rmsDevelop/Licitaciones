#Elabora los seis modelos {licitaciones,menores} x {num_ofertas,zero_discount,
#discount} de forma secuencial, con GPU cuando este disponible. Recetas = los
#campeones de Licitaciones-Lab (BUILDLOG sesion 9; specs exactas en
#experimentos/*.trainings.json). Este archivo NO evalua (la evaluacion es
#evaluate.py sobre filas test) ni enruta (el gate zero_discount->0 lo decide
#quien sirva). Su unica salida son los boosters Models/<linea>.ubj + .meta.json.
#
#Receta (puerto del protocolo del Lab):
#   1. poblacion de la linea: filas train de features.parquet con el target
#      valido (num_ofertas 0..50 o discount_pct 0..70 segun linea) — reproduce
#      las ok_* del Lab;
#   2. split cronologico train/validation: VAL = ultimos 6 meses, solo como
#      ventana de early-stopping para congelar los rounds;
#   3. encoding del organo POR LINEA (decision del Lab): TE suavizado
#      (te/te_extra), frecuencia de train o categorica nativa;
#   4. sonda con early-stopping (30 rondas, como el Lab): el TE de TRAIN va
#      out-of-fold (K=5, seed 42) — sin fuga; rounds = best_iteration + 1;
#   5. booster de produccion: refit con TODAS las filas (train+val) y rounds
#      congelados; el mapa TE/frecuencia se re-fit sobre train+val y ES el que
#      va al meta (consistencia booster<->servido);
#   6. Models/<linea>.ubj + .meta.json (contrato de Inference/: features,
#      niveles categoricos, encoding_organo, transform y clip).
#
#Divergencias documentadas respecto del Lab: aqui el refit de produccion
#existe (el Lab no guarda artefactos) y su mapa TE incluye las filas de val
#(auto-inclusion acotada por el suavizado; los rounds se congelaron con la
#sonda OOF, sin fuga); GPU uniforme (el campeon de men/num corro en CPU con
#un desplazamiento de +0,2-0,4% en MAE documentado en el Lab).

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

import featurer

# ---------------------------------------------------------------------------
# Configuracion de las seis lineas (campeones de Licitaciones-Lab)
# ---------------------------------------------------------------------------
ORGANO = "f_organo_contratante"     # la columna que codifica cada linea a su modo
ORGANO_TE = "f_organo_te"           # columna sintetica del encoding te_extra
TE_K = 5                            # pliegues OOF de la sonda (seed abajo)
SEED = 42
N_JOBS = 6
EARLY_STOPPING = 30                 # rondas de paciencia, como el Lab
ROUNDS_BUFFER = 0                   # rounds = best_iteration + 1 exacto
VAL_MESES = 6                       # ventana de early-stopping

# encoding: "te" (reemplaza el organo por su TE) · "te_extra" (columna
# f_organo_te adicional, organo nativo conservado) · "frecuencia" (reemplaza,
# sin fill) · "nativo" (categorica tal cual).
LINEAS = [
    dict(linea="licitaciones/num_ofertas", tipo="reg",
         objetivo="reg:absoluteerror", eval_metric="mae",
         encoding="te", te_m=100.0, max_depth=8, eta=0.1, rounds_max=1500,
         clip=(0.0, 50.0)),                       # xgb_d8_eta01_mae_teorg_m100_t3
    dict(linea="licitaciones/zero_discount", tipo="clf",
         objetivo="binary:logistic", eval_metric="auc",
         encoding="frecuencia", max_depth=8, eta=0.1, rounds_max=1200),
                                                  # xgb_d8_eta01_freqorg_t3
    dict(linea="licitaciones/discount", tipo="reg",
         objetivo="reg:absoluteerror", eval_metric="mae",
         encoding="te", te_m=20.0, max_depth=8, eta=0.1, rounds_max=1500,
         clip=(0.0, 70.0)),                       # xgb_d8_eta01_mae_teorg_m20_clip_t3
    dict(linea="menores/num_ofertas", tipo="reg",
         objetivo="reg:absoluteerror", eval_metric="mae",
         encoding="te_extra", te_m=20.0, max_depth=8, eta=0.3, rounds_max=600,
         clip=(0.0, 50.0)),                       # xgb_d8_mae_teorg_m20nat
    dict(linea="menores/zero_discount", tipo="clf",
         objetivo="binary:logistic", eval_metric="auc",
         encoding="nativo", max_depth=6, eta=0.3, rounds_max=600),
                                                  # xgb_d6_t3
    dict(linea="menores/discount", tipo="reg",
         objetivo="reg:absoluteerror", eval_metric="mae",
         encoding="te", te_m=20.0, max_depth=10, eta=0.3, rounds_max=600,
         clip=(0.0, 70.0)),                       # xgb_d10_mae_teorg_m20_t3
]


# ---------------------------------------------------------------------------
# Encoding del organo (puerto de _organo_te/_organo_frecuencia del Lab)
# ---------------------------------------------------------------------------
def mapa_te(niveles: pd.Series, y: np.ndarray, m: float) -> tuple[dict, float]:
    """Mapa TE suavizado fit sobre las filas dadas + el prior (media de y).

    te(g) = (n_g*media_g + m*prior) / (n_g + m) — igual formula que el Lab.
    """
    prior = float(np.mean(y))
    g = pd.DataFrame({"lvl": niveles.to_numpy(), "y": y}).groupby("lvl")["y"]
    agg = g.agg(["sum", "count"])
    te = (agg["sum"] + m * prior) / (agg["count"] + m)
    return {k: float(v) for k, v in te.items()}, prior


def aplicar_te(niveles: pd.Series, mapa: dict, prior: float) -> np.ndarray:
    return niveles.map(mapa).fillna(prior).to_numpy(dtype="float64")


def aplicar_te_oof(niveles_tr: pd.Series, y_tr: np.ndarray, m: float,
                   k: int = TE_K) -> tuple[np.ndarray, dict, float]:
    """Valores TE out-of-fold para las filas de train (K pliegues, seed fija):
    la fila nunca ve su propio grupo. Sin mapa en su pliegue -> prior."""
    mapa_full, prior = mapa_te(niveles_tr, y_tr, m)
    folds = np.random.default_rng(SEED).permutation(len(y_tr)) % k
    val = np.full(len(y_tr), prior, dtype="float64")
    for f in range(k):
        en_f = folds == f
        mapa_f, prior_f = mapa_te(niveles_tr[~en_f].reset_index(drop=True),
                                  y_tr[~en_f], m)
        val[en_f] = (niveles_tr[en_f].map(mapa_f).fillna(prior_f)
                     .to_numpy(dtype="float64"))
    return val, mapa_full, prior


def mapa_frecuencia(niveles: pd.Series) -> dict:
    """Frecuencia relativa de cada nivel; no vistos/nulos -> NaN (sin fill)."""
    return {k: float(v) for k, v in niveles.value_counts(normalize=True).items()}


def codificar_organo_sonda(X: pd.DataFrame, y: pd.Series, tr: np.ndarray,
                           val: np.ndarray, linea: dict) -> None:
    """Encoding del organo de la SONDA, in place: TRAIN out-of-fold, VAL con
    el mapa full-train (la replica exacta del regimen del Lab).

    te/frecuencia dejan la columna ORGANO numerica (reemplaza); te_extra
    anade ORGANO_TE y conserva la cruda; nativo no toca nada.
    """
    tipo = linea["encoding"]
    if tipo == "nativo":
        return
    crudos = X[ORGANO].astype("string")
    idx_tr, idx_val = X.index[tr], X.index[val]
    if tipo == "frecuencia":
        mapa = mapa_frecuencia(crudos[idx_tr])
        serie = pd.Series(np.nan, index=X.index, dtype="float64")
        serie.loc[idx_tr] = crudos[idx_tr].map(mapa).to_numpy(dtype="float64")
        serie.loc[idx_val] = crudos[idx_val].map(mapa).to_numpy(dtype="float64")
        X[ORGANO] = serie.to_numpy()
        return
    oof, mapa, prior = aplicar_te_oof(crudos[idx_tr].reset_index(drop=True),
                                      y[tr].to_numpy(dtype=float), linea["te_m"])
    dest = ORGANO_TE if tipo == "te_extra" else ORGANO
    serie = pd.Series(np.nan, index=X.index, dtype="float64")
    serie.loc[idx_tr] = oof
    serie.loc[idx_val] = aplicar_te(crudos[idx_val], mapa, prior)
    X[dest] = serie.to_numpy()


def codificar_organo_produccion(X: pd.DataFrame, y: pd.Series,
                                linea: dict) -> dict:
    """Encoding del organo del booster de PRODUCCION, in place: mapa re-fit
    sobre TODAS las filas (train+val) — el mismo que ira al meta, consistencia
    booster<->servido. Devuelve el bloque encoding_organo del meta."""
    tipo = linea["encoding"]
    if tipo == "nativo":
        return {"tipo": "nativo", "columna": ORGANO}
    crudos = X[ORGANO].astype("string")
    if tipo == "frecuencia":
        mapa = mapa_frecuencia(crudos)
        X[ORGANO] = crudos.map(mapa).to_numpy(dtype="float64")
        return {"tipo": "frecuencia", "columna": ORGANO, "maps": mapa}
    mapa, prior = mapa_te(crudos, y.to_numpy(dtype=float), linea["te_m"])
    dest = ORGANO_TE if tipo == "te_extra" else ORGANO
    X[dest] = aplicar_te(crudos, mapa, prior)
    return {"tipo": tipo, "columna": ORGANO, "m": linea["te_m"],
            "prior": prior, "maps": mapa}


# ---------------------------------------------------------------------------
# Matrices de la linea
# ---------------------------------------------------------------------------
def preparar_matrices(df: pd.DataFrame, linea: dict):
    """Poblacion de la linea + split cronologico + X/y con la capa cruda.

    Devuelve (X, y, ym, tr, val): el encoding del organo (que depende del
    split) lo aplica entrenar_linea por separado en sonda y produccion.
    """
    target_name = linea["linea"].split("/")[1]
    target = featurer.TARGET_LINEA[target_name][0]

    if target_name == "num_ofertas":
        pob = df["num_ofertas"].notna() & df["num_ofertas"].between(0, 50)
    else:
        pob = df["discount_pct"].notna() & df["discount_pct"].between(0, 70)
    sub = df[pob.to_numpy()].reset_index(drop=True)

    ym = sub["fecha_publicacion"].dt.to_period("M")
    val_end = ym.max()
    val_start = val_end - (VAL_MESES - 1)
    val = ((ym >= val_start) & (ym <= val_end)).to_numpy(dtype=bool)
    tr = ~val

    feats = list(featurer.FEATURES_LINEA[linea["linea"]])
    X = sub[feats].copy()             # ORGANO_TE no existe en el parquet: la
    if linea["encoding"] == "te_extra":   # crea el encoding, al final (Lab)
        X[ORGANO_TE] = np.nan
    y = sub[target].astype(int if linea["tipo"] == "clf" else float)
    return X, y, ym, tr, val


def castear_categoricas(X: pd.DataFrame, tr: np.ndarray, linea: dict,
                        niveles_fijos: dict[str, list[str]] | None = None
                        ) -> dict[str, list[str]]:
    """Categoricas nativas: niveles fit-on-train por orden de aparicion (como
    el Lab); codigo -1 (no visto en train o nulo) -> NaN.

    Con niveles_fijos usa esos niveles (la produccion re-castea con los
    niveles de la sonda: los del meta, el contrato de servido). El organo solo
    entra si sigue siendo crudo (nativo/te_extra). Devuelve los niveles.
    """
    skip = {ORGANO_TE}
    if linea["encoding"] in ("te", "frecuencia"):
        skip.add(ORGANO)              # ya numerica tras el encoding
    cat_levels: dict[str, list[str]] = {}
    for c in X.columns:
        if c in skip or featurer.TIPOS.get(c) != "cat":
            continue
        s = X[c].astype("string")
        if niveles_fijos is not None and c in niveles_fijos:
            niveles = pd.Index(niveles_fijos[c])
        else:
            niveles = pd.Index(s[tr].dropna().unique())
        cat_levels[c] = niveles.tolist()
        codes = s.map({v: i for i, v in enumerate(niveles)}).astype("Int32")
        arr = codes.to_numpy(dtype="int32", na_value=-1)
        X[c] = pd.Categorical.from_codes(arr, dtype=pd.CategoricalDtype(niveles))
    return cat_levels


def castear_numericas(X: pd.DataFrame) -> None:
    """Numericas a float64 con NA -> NaN (f_es_pyme boolean -> {0,1,NaN})."""
    for c in X.columns:
        if str(X[c].dtype) == "category":
            continue
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("boolean").astype("Float64")
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")


def hacer_xgb(linea: dict, n_estimators: int, device: str,
              early: int | None = None):
    kw = dict(objective=linea["objetivo"], eval_metric=linea["eval_metric"],
              max_depth=linea["max_depth"], learning_rate=linea["eta"],
              n_estimators=n_estimators, tree_method="hist",
              enable_categorical=True, device=device,
              random_state=SEED, n_jobs=N_JOBS)
    if early is not None:
        kw["early_stopping_rounds"] = early
    cls = XGBClassifier if linea["tipo"] == "clf" else XGBRegressor
    return cls(**kw)


# ---------------------------------------------------------------------------
# Una linea completa
# ---------------------------------------------------------------------------
def entrenar_linea(df: pd.DataFrame, linea: dict, models_dir: Path,
                   device_pref: str) -> dict:
    """Entrena una linea y guarda booster + meta. No evalua, no enruta."""
    t0 = time.time()
    nombre = linea["linea"]
    print(f"\n=== {nombre} ===")

    X, y, ym, tr, val = preparar_matrices(df, linea)
    print(f"poblacion {len(X):,} | TRAIN {int(tr.sum()):,} | "
          f"early-stop VAL {ym[val].min()}..{ym[val].max()} ({int(val.sum()):,}) | "
          f"features {X.shape[1]} | organo: {linea['encoding']}")

    # Sonda: encoding sin fuga (OOF en train) + categoricas fit-on-train ->
    # rounds congelados. Los niveles quedan congelados para produccion/meta.
    Xs = X.copy()
    codificar_organo_sonda(Xs, y, tr, val, linea)
    cat_levels = castear_categoricas(Xs, tr, linea)
    castear_numericas(Xs)

    device = device_pref
    try:
        sonda = hacer_xgb(linea, linea["rounds_max"], device, early=EARLY_STOPPING)
        sonda.fit(Xs[tr], y[tr].to_numpy(), eval_set=[(Xs[val], y[val].to_numpy())],
                  verbose=False)
    except Exception as e:  # noqa: BLE001 — GPU sin memoria u otro fallo -> CPU
        if device_pref == "cpu":
            raise
        print(f"  ({type(e).__name__}: {e})\n  reintentando en CPU")
        device = "cpu"
        sonda = hacer_xgb(linea, linea["rounds_max"], device, early=EARLY_STOPPING)
        sonda.fit(Xs[tr], y[tr].to_numpy(), eval_set=[(Xs[val], y[val].to_numpy())],
                  verbose=False)
    rounds = int(sonda.best_iteration) + 1 + ROUNDS_BUFFER
    del sonda, Xs
    gc.collect()
    print(f"rounds congelados: {rounds}")

    # Booster de produccion: encoding re-fit sobre train+val (el mapa del
    # meta) + categoricas con los niveles congelados de la sonda.
    Xp = X.copy()
    enc_organo = codificar_organo_produccion(Xp, y, linea)
    castear_categoricas(Xp, tr, linea, niveles_fijos=cat_levels)
    castear_numericas(Xp)

    produccion = hacer_xgb(linea, rounds, device)
    produccion.fit(Xp, y.to_numpy(), verbose=False)

    # Guardar booster + meta (contrato de Inference/).
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = nombre.replace("/", "_")
    produccion.get_booster().save_model(str(models_dir / f"{stem}.ubj"))
    meta = {
        "linea": nombre,
        "creado": datetime.now(timezone.utc).isoformat(),
        "target": featurer.TARGET_LINEA[nombre.split("/")[1]][0],
        "transform": None,
        "clip": list(linea["clip"]) if linea.get("clip") else None,
        "hp": {
            "objective": linea["objetivo"], "eval_metric": linea["eval_metric"],
            "max_depth": linea["max_depth"], "learning_rate": linea["eta"],
            "n_estimators": rounds, "tree_method": "hist", "device": device,
            "random_state": SEED, "n_jobs": N_JOBS,
            "early_stopping_rounds": EARLY_STOPPING,
        },
        "rounds": rounds,
        "early_stop_val_meses": VAL_MESES,
        "early_stop_val_window": [str(ym[val].min()), str(ym[val].max())],
        "sizes": {"train": int(tr.sum()), "val": int(val.sum()), "total": int(len(X))},
        "features": list(Xp.columns),
        "categorical_levels": cat_levels,
        "dtype_rule": "numericas float64 (f_es_pyme bool -> {0,1,NaN}); categoricas "
                      "fit-on-train por orden de aparicion, codigo -1 (no visto/"
                      "nulo) -> NaN",
        "encoding_organo": enc_organo,
        "device": device,
    }
    (models_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"guardado: {stem}.ubj (+ meta, {rounds} rounds, {time.time() - t0:.0f}s)")

    del X, Xp
    gc.collect()
    return {"meta": meta}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(data_dir: Path, models_dir: Path, solo: str | None, device: str) -> None:
    feats_path = data_dir / "features.parquet"
    df_cache: dict[str, pd.DataFrame] = {}
    for linea in LINEAS:
        if solo and solo not in linea["linea"]:
            continue
        conjunto = linea["linea"].split("/")[0]
        if conjunto not in df_cache:
            print(f"cargando features de {conjunto} ...")
            df_cache[conjunto] = pd.read_parquet(feats_path, filters=[("conjunto", "=", conjunto)])
        entrenar_linea(df_cache[conjunto], linea, models_dir, device)
        if linea["linea"].endswith("/discount"):
            del df_cache[conjunto]
            gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Elaborar los seis modelos (campeones de Licitaciones-Lab).")
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--models-dir", default="Models")
    p.add_argument("--solo", default=None, help="entrenar solo lineas que contengan este substring")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    run(Path(args.data_dir), Path(args.models_dir), args.solo, args.device)


if __name__ == "__main__":
    main()
