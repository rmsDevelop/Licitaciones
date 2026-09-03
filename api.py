#La API de operacion: estado del parquet, metas de los modelos expuestos,
#evaluaciones registradas por Inference/evaluate.py y el servido a demanda.
#Adaptada de ../Licitaciones/spts/api.py a este rework: sin registry de
#boosters (el servido es a demanda por diseno, Inference/inference.py carga
#y descarga), sin config files, y con un unico parquet en Data/ como almecen.
#
#Endpoints:
#  GET  /                el dashboard (Dashboard/dashboard.html, un solo
#                        fichero, sin build step)
#  GET  /status          universo del parquet + version expuesta + op en
#                        marcha + cola de runs
#  GET  /datos           analisis del parquet (paneles de Datos) — agregados
#                        cacheados por mtime: el poll del dashboard no relee
#                        los 600 MB
#  GET  /modelos         metas de las 6 lineas expuestas + umbrales del router
#                        (nunca los boosters)
#  GET  /evaluaciones    estado_curso.json (modelo expuesto) + historico.jsonl
#                        (modelos ya reemplazados)
#  POST /ops/evaluar     lanza Inference/evaluate.py en subprocess (un solo
#                        op a la vez); ?modo=curso|prepromote
#  POST /inferir         EL producto: parquet raw (+columna 'conjunto') ->
#                        mismo parquet con las 5 preds + version (inferir()
#                        de Inference/inference.py, passthrough completo)
#
#Los ops corren como SUBPROCESOS con log en Data/ops/logs/ y una linea por
#run en Data/ops/runs.jsonl (gitignored como todo Data/). Lock single-flight:
#un op a la vez; el estado en marcha vive en memoria del proceso (si la API
#muere con un op lanzado, el subprocess sigue y su log queda en disco).
#
#Contrato de servicio: LOCALHOST POR DISENO — exposicion, auth y TLS belong
#al sistema que se ponga delante.

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data"
MODELS_DIR = ROOT / "Models"
FEATS_PATH = DATA_DIR / "features.parquet"
DASHBOARD = ROOT / "Dashboard" / "dashboard.html"

sys.path.insert(0, str(ROOT / "Inference"))
import inference  # noqa: E402  (version_modelos, UMBRAL_ZERO_DISCOUNT, inferir)

CONJUNTOS = ("licitaciones", "menores")
LINEAS = ("num_ofertas", "zero_discount", "discount")
MAX_SUBIDA_BYTES = 2**28  # 256 MB: el parquet de un upload de servido
OPS_DIR = DATA_DIR / "ops"

_AGG_CACHE: dict = {}   # agregados del parquet (firma: mtime+tamano)
_META_CACHE: dict = {}  # version expuesta (firma: mtimes de los 6 metas)


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Sin warm-up de boosters: el servido a demanda es el diseno (BUILDLOG
    # Inference/inference.py); el primer /inferir paga la carga (~14 s).
    yield


app = FastAPI(title="nuevo_licitaciones", version="0.1.0", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Cacheos por mtime: el dashboard pollea /status cada 10 s
# ---------------------------------------------------------------------------
def _cache(sig, cache: dict, fn):
    """fn() cacheado mientras la firma (mtimes) no cambie."""
    if cache.get("sig") != sig:
        cache["sig"] = sig
        cache["valor"] = fn()
    return cache["valor"]


def _agregados() -> dict:
    """Analisis del parquet en memoria: universo, filas por ano, calidad.

    Lee solo las columnas del analisis (no los 600 MB enteros) y cachea por
    (mtime, tamano): releer solo cuando update.py reescribe el almecen.
    """
    p = DATA_DIR / "licitaciones.parquet"
    st = p.stat()

    def calc() -> dict:
        import pyarrow.parquet as pq

        n_cols = pq.ParquetFile(p).metadata.num_columns
        df = pd.read_parquet(p, columns=["conjunto", "ml_estado", "version",
                                         "fecha_publicacion", "score_calidad"])
        conjuntos, por_ano = {}, {}
        for c in CONJUNTOS:
            g = df[df["conjunto"] == c]
            est = g["ml_estado"]
            conjuntos[c] = {
                "train": int((est == "train").sum()),
                "test": int((est == "test").sum()),
                "filtered": int((est == "filtered").sum()),
                "abiertas": int(est.isna().sum()),
                "servidas": int(g["version"].notna().sum()),
                "calidad_media": round(float(g["score_calidad"].mean()), 3),
            }
            # filas por ano de publicacion; las pre-2021 (fuera de la ventana
            # de scrape pero presentes en los ZIPs) en un bucket propio
            ano = g["fecha_publicacion"].dt.year
            tempranas = int((ano < 2021).sum())
            serie = (ano[ano >= 2021].astype("Int64").astype(str)
                     .value_counts().sort_index())
            por_ano[c] = ([["<2021", tempranas]] if tempranas else []) + [
                [k, int(v)] for k, v in serie.items()]
        return {
            "filas": len(df), "columnas": n_cols,
            "mb": round(st.st_size / 1e6, 1),
            "modificado": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "conjuntos": conjuntos,
            "filas_por_ano": por_ano,
            "versiones_pred": {f"v{k}": int(v) for k, v
                               in df["version"].value_counts().items()},
        }

    return _cache((st.st_mtime_ns, st.st_size), _AGG_CACHE, calc)


_META_CACHE: dict = {}


def _version_expuesta() -> str:
    """max(meta['creado']) de las 6 lineas (inference.version_modelos),
    cacheado por mtime de los metas: no reparsear ~6 MB de JSON cada poll."""
    sig = tuple((MODELS_DIR / f"{s}.meta.json").stat().st_mtime_ns
                for stems in inference.STEMS.values() for s in stems)
    return _cache(sig, _META_CACHE, lambda: inference.version_modelos(MODELS_DIR))


def _rec_linea(stem: str) -> dict:
    """Resumen del meta de una linea para el panel del modelo expuesto."""
    meta = json.loads((MODELS_DIR / f"{stem}.meta.json").read_text())
    hp = meta["hp"]
    return {
        "linea": meta["linea"], "creado": meta["creado"], "target": meta["target"],
        "transform": meta["transform"], "clip": meta["clip"],
        "hp": {k: hp.get(k) for k in ("objective", "eval_metric", "max_depth",
                                      "learning_rate", "subsample",
                                      "colsample_bytree", "min_child_weight",
                                      "scale_pos_weight", "n_estimators")},
        "rounds": meta["rounds"], "val_window": meta["early_stop_val_window"],
        "sizes": meta["sizes"], "n_features": len(meta["features"]),
        "te": meta.get("target_encoding") is not None,
        "tam_ubj_mb": round((MODELS_DIR / f"{stem}.ubj").stat().st_size / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Ops en subprocess (single-flight)
# ---------------------------------------------------------------------------
_OPS_LOCK = threading.Lock()
_OPS_STATE: dict = {"op": None, "run_id": None, "started": None, "rc": None, "log": None}


def _start_op(op: str, cmd: list[str]) -> dict:
    """Lanza el subprocess del op bajo el lock; no bloqueante."""
    with _OPS_LOCK:
        if _OPS_STATE["op"] is not None:
            raise HTTPException(409, f"ya hay un op en marcha: {_OPS_STATE['op']}")
        run_id = time.strftime("%Y%m%d_%H%M%S")
        started = _ahora()
        log = OPS_DIR / "logs" / f"{op}_{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        _OPS_STATE.update(op=op, run_id=run_id, started=started, rc=None, log=str(log))

    proc = subprocess.Popen(cmd, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                            cwd=ROOT)

    def _watch():
        rc = proc.wait()
        with _OPS_LOCK:
            rec = {"op": op, "run_id": run_id, "triggered": "api",
                   "started": _OPS_STATE["started"], "finished": _ahora(),
                   "status": "ok" if rc == 0 else "failed", "rc": rc}
            with open(OPS_DIR / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            _OPS_STATE.update(op=None, rc=rc)

    threading.Thread(target=_watch, daemon=True).start()
    return {"op": op, "run_id": run_id, "log": str(log), "cmd": " ".join(cmd)}


def _ops_estado() -> dict:
    with _OPS_LOCK:
        return dict(_OPS_STATE)


def _runs_tail(n: int = 12) -> list[dict]:
    p = DATA_DIR / "ops" / "runs.jsonl"
    if not p.exists():
        return []
    lineas = p.read_text().splitlines()[-n:]
    return [json.loads(l) for l in reversed(lineas) if l.strip()]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/status")
def status() -> dict:
    return {
        "version": _version_expuesta(),
        "universo": {c: _agregados()["conjuntos"][c] for c in CONJUNTOS},
        "ops": _ops_estado(),
        "runs_tail": _runs_tail(),
    }


@app.get("/datos")
def datos() -> dict:
    return _agregados()


@app.get("/modelos")
def modelos() -> dict:
    """Las 6 lineas expuestas ahora mismo: identidad + receta de entrenamiento.

    Lee los metas (nunca los boosters). No hay metricas de entrega en el meta
    (el entrenamiento del rework no evalua: la evaluacion es evaluate.py sobre
    filas test) — el panel vive de /evaluaciones.
    """
    lineas = {c: {} for c in CONJUNTOS}
    for c in CONJUNTOS:
        for linea, stem in zip(LINEAS, inference.STEMS[c]):
            lineas[c][linea] = _rec_linea(stem)
    return {"version": _version_expuesta(),
            "umbrales": inference.UMBRAL_ZERO_DISCOUNT,
            "lineas": lineas}


@app.get("/evaluaciones")
def evaluaciones() -> dict:
    """Lo registrado por Inference/evaluate.py en Data/evaluaciones/:
    curso = estado_curso.json del modelo expuesto (se sobreescribe),
    historico = una linea por cierre prepromote de modelos ya reemplazados."""
    d = DATA_DIR / "evaluaciones"
    curso = None
    if (d / "estado_curso.json").exists():
        curso = json.loads((d / "estado_curso.json").read_text())
    historico = []
    if (d / "historico.jsonl").exists():
        for l in (d / "historico.jsonl").read_text().splitlines():
            if l.strip():
                historico.append(json.loads(l))
    return {"curso": curso, "historico": historico}


@app.post("/ops/evaluar")
def ops_evaluar(modo: str = "curso"):
    """Evaluar el sistema expuesto sobre las filas test ahora mismo.

    curso (default): refresca el estado_curso.json del dashboard. prepromote:
    la evaluacion de cierre del modelo que va a ser reemplazado — pertenece
    al protocolo de reentrenamiento, se expone por simetria con el CLI.
    """
    if modo not in ("curso", "prepromote"):
        raise HTTPException(400, "modo debe ser curso|prepromote")
    cmd = [sys.executable, "-u", "Inference/evaluate.py", "--modo", modo]
    return _start_op("evaluar", cmd)


@app.post("/inferir")
async def inferir(file: UploadFile = File(...)):
    """Parquet raw (+columna 'conjunto') -> mismo parquet + 5 preds + version.

    Passthrough completo: el artefacto subido vuelve con las columnas de
    inferencia anadidas. Todo el servido es a demanda (sin cache de boosters).
    """
    data = await file.read()
    if len(data) > MAX_SUBIDA_BYTES:
        raise HTTPException(413, f"la subida excede {MAX_SUBIDA_BYTES} bytes")
    try:
        df = pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(422, f"parquet ilegible: {e}")
    if df.empty:
        # inferir() agrupa por conjunto: con 0 filas el concat final reventaria
        raise HTTPException(422, "el parquet no tiene filas")
    try:
        res = inference.inferir(df, MODELS_DIR, FEATS_PATH)
    except ValueError as e:
        raise HTTPException(409, str(e))
    buf = io.BytesIO()
    res.to_parquet(buf, index=False, compression="snappy")
    nombre = Path(file.filename or "subida.parquet").stem + "_preds.parquet"
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition":
                             f'attachment; filename="{nombre}"',
                             "X-Modelo-Version": res["version"].iloc[0],
                             "X-Filas": str(len(res)),
                             "X-Caveat": "hist-volume-base-tabla-train"})


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD.read_text())
