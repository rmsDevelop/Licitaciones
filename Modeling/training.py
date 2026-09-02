#Entrena las seis lineas {licitaciones,menores} x {num_ofertas,zero_discount,
#discount} en cadena y secuencial, con GPU cuando este disponible. HPs fijados
#a los campeones de Licitaciones-Lab (recipes de los 05_training.py). Receta
#de entrega (la del notebook 06 del Lab, adaptada a que aqui ya no hay TEST
#porque la evaluacion previa ya ocurrio):
#   1. poblacion de la linea: filas train de features.parquet con el target
#      valido (num_ofertas 0..50 o discount_pct 0..70 segun linea);
#   2. split cronologico train/validation: VAL = ultimos 6 meses (la ventana
#      de calibracion del contrato one-fit), TRAIN = el resto hasta 2021-01;
#   3. target encoding (solo lineas campeonas con TE): mapas fit en TRAIN;
#   4. SONDA con early-stopping TRAIN vs VAL -> rounds congelados (+buffer 50)
#      y metricas honestas de VAL (la sonda nunca vio VAL); para zero_discount
#      tambien el umbral recall@precision>=floor;
#   5. booster de produccion: refit con TODAS las filas de la linea y los
#      rounds congelados (lo que servira Inference/);
#   6. metricas de sistema (discount): gate duro (p_zero >= umbral -> 0) con
#      las probs de VAL de la sonda zero_discount de su conjunto;
#   7. Models/<linea>.ubj + Models/<linea>.meta.json (contrato de Inference/:
#      features, niveles categoricos, mapas TE, umbral, transforms, clips).

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, log_loss,
    mean_absolute_error, precision_recall_curve, precision_score, r2_score,
    recall_score, roc_auc_score, root_mean_squared_error,
)
from xgboost import XGBClassifier, XGBRegressor

import featurer

# ---------------------------------------------------------------------------
# Configuracion de las seis lineas (campeones de Licitaciones-Lab)
# ---------------------------------------------------------------------------
TE_COLS = ["organo_contratante", "dir3_organo", "ciudad_organo"]
TE_ALPHA = 30.0
EARLY_STOPPING = 50
ROUNDS_BUFFER = 50
VAL_MESES = 6                      # ventana de calibracion (contrato one-fit)

HP_BASE = dict(subsample=0.8, colsample_bytree=0.8, min_child_weight=5.0,
               reg_lambda=1.0, tree_method="hist", enable_categorical=True,
               random_state=42)

LINEAS = [
    dict(linea="licitaciones/num_ofertas", tipo="reg",
         objetivo="reg:squarederror", eval_metric="mae", transform="log1p",
         te=True, max_depth=13, learning_rate=0.03, n_est_max=1500,
         clip=(0.0, 50.0)),
    dict(linea="licitaciones/zero_discount", tipo="clf",
         objetivo="binary:logistic", eval_metric="aucpr",
         spw="auto", max_depth=14, learning_rate=0.05, n_est_max=1000,
         floor=0.80),
    dict(linea="licitaciones/discount", tipo="reg",
         objetivo="reg:pseudohubererror", eval_metric="mae",
         te=False, max_depth=14, learning_rate=0.03, subsample=0.9,
         n_est_max=1500, clip=(0.0, 70.0)),
    dict(linea="menores/num_ofertas", tipo="reg",
         objetivo="reg:squarederror", eval_metric="mae", transform="log1p",
         te=True, max_depth=13, learning_rate=0.03, n_est_max=1500,
         clip=(0.0, 50.0)),
    dict(linea="menores/zero_discount", tipo="clf",
         objetivo="binary:logistic", eval_metric="aucpr",
         spw="none", max_depth=13, learning_rate=0.03, n_est_max=1000,
         floor=0.95),
    dict(linea="menores/discount", tipo="reg",
         objetivo="reg:pseudohubererror", eval_metric="mae",
         te=True, max_depth=14, learning_rate=0.03, n_est_max=1500,
         clip=(0.0, 70.0)),
]


# ---------------------------------------------------------------------------
# Trozos de receta (puerto de los 05_training.py del Lab)
# ---------------------------------------------------------------------------
def resolver_spw(val: str, prevalencia: float) -> float:
    v = str(val).strip().lower()
    if v in ("auto", "balanced", "weighted"):
        pos = max(float(prevalencia), 1e-9)
        return round((1 - pos) / pos, 3)
    if v in ("none", "unweighted", "1", "1.0"):
        return 1.0
    return float(v)


def elegir_umbral(y_true, p_prob, floor: float) -> float:
    """Umbral que maximiza recall con precision >= floor (fallback F1-max)."""
    prec, rec, thr = precision_recall_curve(y_true, p_prob)
    prec, rec = prec[:-1], rec[:-1]
    ok = prec >= floor
    if ok.any():
        return float(thr[ok][np.argmax(rec[ok])])
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1)])


def add_target_encodings(X: pd.DataFrame, y: pd.Series, train_mask: np.ndarray,
                         alpha: float = TE_ALPHA) -> tuple[pd.DataFrame, dict]:
    """Encoding suavizado de las categoricas de alta cardinalidad.

    Mapas fit en TRAIN (cronologicamente previos) -> sin fuga. Los niveles
    raros se encogen hacia la media global de TRAIN via alpha. Los mapas se
    guardan en el meta para aplicarlos identicos en inferencia.
    """
    X = X.copy()
    ytr = y[train_mask].astype(float)
    gmean = float(ytr.mean())
    te_maps = {}
    for col in TE_COLS:
        lvl_all = X[col].astype("string")
        lvl_tr = X.loc[train_mask, col].astype("string")
        agg = (pd.DataFrame({"lvl": lvl_tr.to_numpy(), "y": ytr.to_numpy()})
               .groupby("lvl")["y"].agg(["mean", "count"]))
        te_map = (agg["mean"] * agg["count"] + alpha * gmean) / (agg["count"] + alpha)
        te_maps[col] = {k: float(v) for k, v in te_map.items()}
        X[col + "_te"] = lvl_all.map(te_map).fillna(gmean).astype("float32")
    return X.drop(columns=TE_COLS), {"gmean": gmean, "alpha": alpha, "maps": te_maps}


def preparar_matrices(df: pd.DataFrame, linea: dict):
    """Poblacion de la linea + X/y + split cronologico + TE + casts.

    Devuelve (X, y, ym, tr, val, ids_val, cat_levels, te_info). Indices
    posicionales (reset) para alinear ids y probs entre lineas sin ambiguedad.
    """
    conjunto, target_name = linea["linea"].split("/")
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

    feats = [c for c in featurer.FEATURES_LINEA[linea["linea"]]
             if c in sub.columns and c != "fecha_publicacion"]
    X = sub[feats].copy()
    y = sub[target].astype(int if linea["tipo"] == "clf" else float)

    te_info = None
    if linea.get("te"):
        X, te_info = add_target_encodings(X, y, tr)

    # Categoricas: niveles ordenados registrados en el meta (contrato al
    # servir: niveles nuevos -> missing). El resto, float32.
    cat_levels = {}
    for c in X.columns:
        if c in featurer.CATEGORICAL_BASE or c == "procedimiento_code":
            X[c] = X[c].astype("string")
            niveles = sorted(X[c].dropna().unique().tolist())
            cat_levels[c] = niveles
            X[c] = pd.Categorical(X[c], categories=niveles)
        elif str(X[c].dtype) != "category":
            X[c] = X[c].astype("float32")

    ids_val = sub.loc[val, "id"].to_numpy()
    return X, y, ym, tr, val, ids_val, cat_levels, te_info


def hacer_xgb(linea: dict, n_estimators: int, device: str,
              spw: float | None = None, early: int | None = None):
    kw = dict(HP_BASE)
    kw.update(objective=linea["objetivo"], eval_metric=linea["eval_metric"],
              max_depth=linea["max_depth"], learning_rate=linea["learning_rate"],
              n_estimators=n_estimators, device=device)
    if "subsample" in linea:
        kw["subsample"] = linea["subsample"]
    if spw is not None:
        kw["scale_pos_weight"] = spw
    if early is not None:
        kw["early_stopping_rounds"] = early
    cls = XGBClassifier if linea["tipo"] == "clf" else XGBRegressor
    return cls(**kw)


def _target_y(y: pd.Series, transform: str | None):
    arr = y.to_numpy(dtype=float)
    return np.log1p(arr) if transform == "log1p" else arr


def metricas_reg(y_true, y_pred_raw, clip, transform=None, por_zeros=False) -> dict:
    yt = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred_raw, dtype=float)
    if transform == "log1p":
        p = np.expm1(p)
    yp = np.clip(p, *clip)
    out = {
        "mae": round(float(mean_absolute_error(yt, yp)), 4),
        "rmse": round(float(root_mean_squared_error(yt, yp)), 4),
        "r2": round(float(r2_score(yt, yp)), 3),
        "bias": round(float(np.mean(yp) - np.mean(yt)), 3),
        "mean_pred": round(float(np.mean(yp)), 3),
    }
    if por_zeros:
        zeros = yt == 0
        out["mae_zero_rows"] = round(float(np.mean(np.abs(yt[zeros] - yp[zeros]))), 3) if zeros.any() else None
        out["mae_positive_rows"] = round(float(np.mean(np.abs(yt[~zeros] - yp[~zeros]))), 3) if (~zeros).any() else None
    return out


def metricas_clf(y_true, p, umbral, floor) -> dict:
    yt = np.asarray(y_true)
    pred = (p >= umbral).astype(int)
    return {
        "auc_pr": round(float(average_precision_score(yt, p)), 3),
        "auc_roc": round(float(roc_auc_score(yt, p)), 3),
        "logloss": round(float(log_loss(yt, p, labels=[0, 1])), 3),
        "precision": round(float(precision_score(yt, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(yt, pred, zero_division=0)), 3),
        "f1": round(float(f1_score(yt, pred, zero_division=0)), 3),
        "accuracy": round(float(accuracy_score(yt, pred)), 3),
        "prevalence": round(float(np.mean(yt)), 3),
        "umbral": round(float(umbral), 4),
    }


# ---------------------------------------------------------------------------
# Una linea completa
# ---------------------------------------------------------------------------
def entrenar_linea(df: pd.DataFrame, linea: dict, models_dir: Path, device_pref: str,
                   zd_val: dict | None) -> dict:
    """Entrena una linea y guarda booster + meta. Devuelve el meta y, para las
    lineas zero_discount, las probs de VAL por id (input del discount)."""
    t0 = time.time()
    nombre = linea["linea"]
    transform = linea.get("transform")
    es_disc = nombre.endswith("/discount")
    print(f"\n=== {nombre} ===")

    X, y, ym, tr, val, ids_val, cat_levels, te_info = preparar_matrices(df, linea)
    print(f"poblacion {len(X):,} | TRAIN {int(tr.sum()):,} | "
          f"VAL {ym[val].min()}..{ym[val].max()} ({int(val.sum()):,}) | "
          f"features {X.shape[1]} ({len(cat_levels)} cat)")

    spw = None
    if linea["tipo"] == "clf":
        spw = resolver_spw(linea["spw"], float(y[tr].mean()))

    # Sonda: early-stop TRAIN vs VAL -> rounds + metricas honestas + umbral.
    device = device_pref
    try:
        sonda = hacer_xgb(linea, linea["n_est_max"], device, spw, early=EARLY_STOPPING)
        if linea["tipo"] == "reg":
            sonda.fit(X[tr], _target_y(y[tr], transform),
                      eval_set=[(X[val], _target_y(y[val], transform))], verbose=False)
        else:
            sonda.fit(X[tr], y[tr].to_numpy(), eval_set=[(X[val], y[val].to_numpy())], verbose=False)
    except Exception as e:  # noqa: BLE001 — GPU sin memoria u otro fallo -> CPU
        if device_pref == "cpu":
            raise
        print(f"  ({type(e).__name__}: {e})\n  reintentando en CPU")
        device = "cpu"
        sonda = hacer_xgb(linea, linea["n_est_max"], device, spw, early=EARLY_STOPPING)
        if linea["tipo"] == "reg":
            sonda.fit(X[tr], _target_y(y[tr], transform),
                      eval_set=[(X[val], _target_y(y[val], transform))], verbose=False)
        else:
            sonda.fit(X[tr], y[tr].to_numpy(), eval_set=[(X[val], y[val].to_numpy())], verbose=False)
    rounds = int(sonda.best_iteration) + 1 + ROUNDS_BUFFER

    umbral, metricas, p_val = None, {}, None
    if linea["tipo"] == "clf":
        p_val = sonda.predict_proba(X[val])[:, 1]
        umbral = elegir_umbral(y[val].to_numpy(), p_val, linea["floor"])
        metricas = metricas_clf(y[val].to_numpy(), p_val, umbral, linea["floor"])
    else:
        pred_val = sonda.predict(X[val])
        metricas = metricas_reg(y[val].to_numpy(), pred_val, linea["clip"],
                                transform, por_zeros=es_disc)
        yt = y[val].to_numpy()
        metricas["baseline_media"] = round(float(mean_absolute_error(yt, np.full(len(yt), y[tr].mean()))), 4)
        metricas["baseline_mediana"] = round(float(mean_absolute_error(yt, np.full(len(yt), y[tr].median()))), 4)
    print(f"sonda ({device}): rounds={rounds} | VAL: {metricas}")

    # Sistema (discount): gate duro con las probs de la sonda zd, por id.
    sistema = None
    if es_disc:
        if zd_val is None:
            raise SystemExit(f"{nombre}: falta la salida de {linea['linea'].split('/')[0]}/zero_discount")
        p_zero = zd_val["probs"].reindex(pd.Index(ids_val)).to_numpy()
        if np.isnan(p_zero).any():
            raise ValueError("las probs del router no cubren todos los ids de VAL")
        pv = sonda.predict(X[val])
        if transform == "log1p":
            pv = np.expm1(pv)
        pv = np.clip(pv, *linea["clip"])
        gate = p_zero >= zd_val["umbral"]
        hard = np.where(gate, 0.0, pv)
        sistema = {
            "system_hard_gate_mae": round(float(mean_absolute_error(y[val].to_numpy(), hard)), 4),
            "router_flag_frac": round(float(gate.mean()), 4),
            "umbral_router": round(float(zd_val["umbral"]), 4),
        }
        print(f"sistema (gate @ {sistema['umbral_router']}): "
              f"MAE={sistema['system_hard_gate_mae']} (router marca {sistema['router_flag_frac']*100:.1f}%)")

    # Booster de produccion: refit con TODAS las filas y rounds congelados.
    produccion = hacer_xgb(linea, rounds, device, spw)
    if linea["tipo"] == "reg":
        produccion.fit(X, _target_y(y, transform), verbose=False)
    else:
        produccion.fit(X, y.to_numpy(), verbose=False)

    # Guardar booster + meta (contrato de Inference/).
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = nombre.replace("/", "_")
    produccion.get_booster().save_model(str(models_dir / f"{stem}.ubj"))
    meta = {
        "linea": nombre,
        "creado": datetime.now(timezone.utc).isoformat(),
        "target": featurer.TARGET_LINEA[nombre.split("/")[1]][0],
        "transform": transform,
        "clip": list(linea["clip"]) if linea.get("clip") else None,
        "umbral": round(float(umbral), 6) if umbral is not None else None,
        "floor_precision": linea.get("floor"),
        "hp": {
            "objective": linea["objetivo"], "eval_metric": linea["eval_metric"],
            "max_depth": linea["max_depth"], "learning_rate": linea["learning_rate"],
            "subsample": linea.get("subsample", HP_BASE["subsample"]),
            "colsample_bytree": HP_BASE["colsample_bytree"],
            "min_child_weight": HP_BASE["min_child_weight"],
            "reg_lambda": HP_BASE["reg_lambda"], "scale_pos_weight": spw,
            "tree_method": "hist", "device": device, "random_state": 42,
            "n_estimators": rounds,
        },
        "rounds": rounds,
        "val_meses": VAL_MESES,
        "sizes": {"train": int(tr.sum()), "val": int(val.sum()), "total": int(len(X))},
        "features": list(X.columns),
        "categorical_levels": cat_levels,
        "dtype_rule": "categoricas segun categorical_levels (NaN/nuevo -> missing); resto float32",
        "target_encoding": te_info,
        "metricas_val_sonda": metricas,
        "sistema": sistema,
        "device": device,
    }
    (models_dir / f"{stem}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"guardado: {stem}.ubj (+ meta, {rounds} rounds, {time.time() - t0:.0f}s)")

    del X
    gc.collect()
    out = {"meta": meta}
    if linea["tipo"] == "clf":
        out["zd_val"] = {"probs": pd.Series(p_val, index=ids_val), "umbral": float(umbral)}
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(data_dir: Path, models_dir: Path, solo: str | None, device: str) -> None:
    feats_path = data_dir / "features.parquet"
    df_cache: dict[str, pd.DataFrame] = {}
    zd_val: dict[str, dict] = {}
    resumen = []
    for linea in LINEAS:
        if solo and solo not in linea["linea"]:
            continue
        conjunto = linea["linea"].split("/")[0]
        if conjunto not in df_cache:
            print(f"cargando features de {conjunto} ...")
            df_cache[conjunto] = pd.read_parquet(feats_path, filters=[("conjunto", "=", conjunto)])
        r = entrenar_linea(df_cache[conjunto], linea, models_dir, device,
                           zd_val.get(conjunto))
        resumen.append(r)
        if "zd_val" in r:
            zd_val[conjunto] = r["zd_val"]
        if linea["linea"].endswith("/discount"):
            zd_val.pop(conjunto, None)
            del df_cache[conjunto]
            gc.collect()

    print("\n=== resumen (metricas honestas de VAL, sonda) ===")
    for r in resumen:
        m, met = r["meta"], r["meta"]["metricas_val_sonda"]
        pr = f"MAE={met['mae']}" if "mae" in met else f"AUCpr={met['auc_pr']}"
        sis = f" | sistema MAE={r['meta']['sistema']['system_hard_gate_mae']}" if r["meta"].get("sistema") else ""
        print(f"{m['linea']:28s} {pr:16s} umbral={m['umbral']}{sis}")


def main() -> None:
    p = argparse.ArgumentParser(description="Entrenar las seis lineas (campeones del Lab).")
    p.add_argument("--data-dir", default="Data")
    p.add_argument("--models-dir", default="Models")
    p.add_argument("--solo", default=None, help="entrenar solo lineas que contengan este substring")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    run(Path(args.data_dir), Path(args.models_dir), args.solo, args.device)


if __name__ == "__main__":
    main()
