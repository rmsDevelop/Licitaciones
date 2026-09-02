#Evalua el sistema expuesto sobre las filas ml_estado=='test' (las adjudicadas
#ajenas al entrenamiento que Scraper/update.py va marcando). Re-infiere el
#conjunto test con los boosters de Models/ (Inference/inference.py) en cada
#evaluacion — nada se persiste en las columnas pred del parquet: esas son el
#registro de servido y las filas test no fueron servidas — y filtra la
#poblacion con los criterios de Modeling/cleaning.py reutilizando su
#clasificador: una fila test solo es evaluable si seria 'train' (ventana y
#objetivo valido); las que no, quedan para que cleaning las reclasifique.
#
#Dos modos de registro en <data-dir>/evaluaciones/ (cada host alberga sus
#datos; Data/ es gitignored):
#   curso      — evaluacion del modelo expuesto en marcha: registro temporal
#                que se sobreescribe (estado_curso.json), para el dashboard
#                del modelo expuesto.
#   prepromote — evaluacion de cierre del modelo que va a ser reemplazado,
#                justo antes de reentrenar: apendice al historico
#                (historico.jsonl). El modelo nuevo cerrara la suya cuando le
#                toque ser reemplazado. Tras registrar, el reentrenamiento
#                (cleaning -> featurer -> training) pliega las test a train.

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Modeling"))
import cleaning

import inference


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def _mae(pred: pd.Series, real: pd.Series) -> float | None:
    if len(pred) == 0:
        return None
    return float(np.mean(np.abs(pred.to_numpy(dtype="float64")
                                - real.to_numpy(dtype="float64"))))


def metricas_conjunto(sub: pd.DataFrame, umbral: float) -> dict:
    """Metricas de las tres lineas + sistema (gate) para un conjunto servido.

    sub lleva las preds de inferir() y las columnas reales num_ofertas y
    discount_pct (derivada por base del conjunto como en cleaning).
    """
    out: dict = {}

    num = sub["num_ofertas"]
    m = (num.notna() & num.between(0, 50)).to_numpy()
    out["num_ofertas"] = {"n": int(m.sum()), "mae": _mae(sub.loc[m, "num_ofertas_pred"], num[m])}

    disc = sub["discount_pct"]
    d = (disc.notna() & disc.between(0, 70)).to_numpy()
    zero = (disc[d].to_numpy(dtype="float64") == 0)
    prob = sub.loc[d, "zero_discount_prob"].to_numpy(dtype="float64")
    zd = prob >= umbral
    tp, fp = int((zd & zero).sum()), int((zd & ~zero).sum())
    fn = int((~zd & zero).sum())
    out["zero_discount"] = {
        "n": int(d.sum()), "prevalencia": float(zero.mean()) if len(zero) else None,
        "aucpr": float(average_precision_score(zero, prob)) if 0 < zero.sum() < len(zero) else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
    out["discount"] = {"n": int(d.sum()), "mae": _mae(sub.loc[d, "discount_pct_pred"], disc[d])}
    out["system"] = {"n": int(d.sum()), "mae": _mae(sub.loc[d, "system_discount_pct_pred"], disc[d])}
    return {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in out.items()}


# ---------------------------------------------------------------------------
# Evaluacion
# ---------------------------------------------------------------------------
def run(data_dir: Path, models_dir: Path, feats_path: Path, modo: str,
        dry_run: bool = False) -> dict:
    path = data_dir / "licitaciones.parquet"
    df = pd.read_parquet(path, filters=[("ml_estado", "==", "test")])
    if df.empty:
        print(f"sin filas test en {path} — nada que evaluar")
        return {}
    print(f"filas test: {len(df):,}")

    # Validez = criterios de cleaning: evaluable si clasificaria como 'train'.
    valido = pd.Series(False, index=df.index)
    for conjunto in ("licitaciones", "menores"):
        mask = df["conjunto"] == conjunto
        valido[mask] = (cleaning.clasificar_estado(df[mask]) == "train").to_numpy()
    print(f"evaluables (serian 'train'): {int(valido.sum()):,} | "
          f"excluidas por invalidas: {int((~valido).sum()):,}")

    sub = df[valido].reset_index(drop=True)
    servido = inference.inferir(sub.drop(columns=["ml_estado"]), models_dir, feats_path)

    # Objetivos reales: num tal cual; discount por base del conjunto (cleaning).
    servido["discount_pct"] = np.nan
    for conjunto in ("licitaciones", "menores"):
        mask = servido["conjunto"] == conjunto
        servido.loc[mask, "discount_pct"] = cleaning.derivar_discount_pct(servido[mask])

    metricas = {}
    for conjunto in ("licitaciones", "menores"):
        c = servido[servido["conjunto"] == conjunto]
        if len(c) == 0:
            continue
        metricas[conjunto] = metricas_conjunto(c, inference.UMBRAL_ZERO_DISCOUNT[conjunto])
        n = metricas[conjunto]
        print(f"  {conjunto}: num MAE {n['num_ofertas']['mae']} (n={n['num_ofertas']['n']}) | "
              f"zd AUC-PR {n['zero_discount']['aucpr']} | "
              f"disc MAE {n['discount']['mae']} | system MAE {n['system']['mae']} "
              f"(n={n['discount']['n']})")

    pub = servido["fecha_publicacion"].dropna()
    registro = {
        "modo": modo,
        "fecha": datetime.now(timezone.utc).isoformat(),
        "version_modelo": inference.version_modelos(models_dir),
        "umbrales_zero_discount": inference.UMBRAL_ZERO_DISCOUNT,
        "filas": {"test": int(len(df)), "evaluadas": int(len(servido)),
                  "excluidas_invalidas": int((~valido).sum())},
        "publicacion": [str(pub.min().date()), str(pub.max().date())] if len(pub) else None,
        "metricas": metricas,
    }

    if dry_run:
        print("(dry-run: no se registra)")
        print(json.dumps(registro, ensure_ascii=False, indent=2))
        return registro

    reg_dir = data_dir / "evaluaciones"
    reg_dir.mkdir(parents=True, exist_ok=True)
    linea = json.dumps(registro, ensure_ascii=False)
    if modo == "curso":
        destino = reg_dir / "estado_curso.json"
        tmp = destino.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(registro, ensure_ascii=False, indent=2))
        os.replace(tmp, destino)
        print(f"registrado (temporal): {destino}")
    else:
        with open(reg_dir / "historico.jsonl", "a") as f:
            f.write(linea + "\n")
        print(f"registrado (historico): {reg_dir / 'historico.jsonl'}")
    return registro


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluar el sistema expuesto sobre las filas ml_estado=='test'.")
    p.add_argument("--modo", choices=["curso", "prepromote"], default="curso",
                   help="curso: registro temporal del modelo expuesto | "
                        "prepromote: evaluacion de cierre del modelo a reemplazar (historico)")
    p.add_argument("--data-dir", default="Data", help="directorio con licitaciones.parquet")
    p.add_argument("--models-dir", default="Models")
    p.add_argument("--feats", default="Data/features.parquet",
                   help="tabla train: base de los hist_volume")
    p.add_argument("--dry-run", action="store_true", help="informe sin registrar")
    args = p.parse_args()
    run(Path(args.data_dir), Path(args.models_dir), Path(args.feats), args.modo, args.dry_run)


if __name__ == "__main__":
    main()
