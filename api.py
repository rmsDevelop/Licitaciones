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
#  GET  /eda             stats basicas + agregados por dimension del parquet
#                        (panel EDA de Datos): ?conjunto=ambos|licitaciones|
#                        menores · medias sobre poblacion evaluable
#  GET  /eda/fig/{dim}   la figura de una dimension (ano|cpv|tipo) como SVG
#                        matplotlib (Dashboard/graficas.py): barras = nº
#                        licitaciones, lineas = descuento y ofertas medias ·
#                        ?conjunto&tema=claro|oscuro · cacheada por mtime
#  GET  /modelos         metas de las 6 lineas expuestas + umbrales del router
#                        (nunca los boosters)
#  GET  /laboratorio     campeones promovidos del Lab (experimentos/ de
#                        ../Licitaciones-Lab) + resumen del registro
#  GET  /evaluaciones    estado_curso.json (modelo expuesto) + historico.jsonl
#                        (modelos ya reemplazados)
#  POST /ops/{op}        lanza un op en subprocess (un solo op a la vez):
#                        evaluar | update | cleaning | featurer | training |
#                        ciclo (protocolo completo encadenado). ?modo=curso|
#                        prepromote (evaluar) · ?meses (update y ciclo)
#  GET  /ops/log/{op}/{id}  ultimas lineas del log de un run (visor)
#  POST /inferir         EL producto: parquet raw (+columna 'conjunto') ->
#                        mismo parquet con las 5 preds + version (inferir()
#                        de Inference/inference.py, passthrough completo)
#
#Los ops corren como SUBPROCESOS con log en Data/ops/logs/ y una linea por
#PASO en Data/ops/runs.jsonl (gitignored como todo Data/); los pasos de un
#ciclo llevan el campo "ciclo" con el id del grupo. Lock single-flight: un
#op a la vez; el estado en marcha vive en memoria del proceso (si la API
#muere con un op lanzado, el subprocess sigue y su log queda en disco).
#El ciclo para al primer paso con rc!=0 — y evaluar con 0 filas test es un
#no-op limpio (rc 0 sin registrar), asi que el ciclo sigue su curso.
#
#Contrato de servicio: LOCALHOST POR DISENO — exposicion, auth y TLS belong
#al sistema que se ponga delante.

from __future__ import annotations

import io
import json
import re
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
DASHBOARD = ROOT / "Dashboard" / "dashboard.html"
# El registro de experimentos del Lab (leido por GET /laboratorio). El Lab NO
# marca campeones en su registro: PROMOVIDOS_LAB es la constante del puerto a
# mano — cada promocion la actualiza.
LAB_DIR = ROOT.parent / "Licitaciones-Lab" / "experimentos"
PROMOVIDOS_LAB = {
    "licitaciones/num_ofertas": "xgb_d8_eta01_mae_teorg_m100_t3",
    "licitaciones/zero_discount": "xgb_d8_eta01_freqorg_t3",
    "licitaciones/discount": "xgb_d8_eta01_mae_teorg_m20_clip_t3",
    "menores/num_ofertas": "xgb_d8_mae_teorg_m20nat",
    "menores/zero_discount": "xgb_d6_t3",
    "menores/discount": "xgb_d10_mae_teorg_m20_t3",
}

sys.path.insert(0, str(ROOT / "Dashboard"))
sys.path.insert(0, str(ROOT / "Inference"))
sys.path.insert(0, str(ROOT / "Modeling"))
import cleaning  # noqa: E402  (DISCOUNT_RANGE, NUM_OFERTAS_RANGE: poblacion evaluable del EDA)
import graficas  # noqa: E402  (fig_eda + TEMAS: el render de /eda/fig)
import inference  # noqa: E402  (version_modelos, UMBRAL_ZERO_DISCOUNT, inferir)

CONJUNTOS = ("licitaciones", "menores")
LINEAS = ("num_ofertas", "zero_discount", "discount")
MAX_SUBIDA_BYTES = 2**28  # 256 MB: el parquet de un upload de servido
OPS_DIR = DATA_DIR / "ops"

# Ops expuestos como subprocess (script; flags fijos + los del query). scrape
# queda fuera (reconstruir el almacen entero es cosa de terminal, no de la
# consola); servir es POST /inferir, no un op.
OPS = {
    "evaluar":  "Inference/evaluate.py",
    "update":   "Scraper/update.py",
    "cleaning": "Modeling/cleaning.py",
    "featurer": "Modeling/featurer.py",
    "training": "Modeling/training.py",
}
# El protocolo de reentrenamiento completo, en orden: evaluar (prepromote)
# cierra el modelo que va a ser reemplazado; cleaning pliega las test a train.
CICLO = ("update", "evaluar", "cleaning", "featurer", "training")

_AGG_CACHE: dict = {}   # agregados del parquet (firma: mtime+tamano)
_META_CACHE: dict = {}  # version expuesta (firma: mtimes de los 6 metas)
_LAB_CACHE: dict = {}   # campeones del Lab (firma: mtimes de los 6 registros)
_EDA_CACHE: dict = {}   # agregados del EDA (firma: mtime+tamano+conjunto)
_EDA_FIG_CACHE: dict = {}  # SVG del EDA por (dim, conjunto, tema) — firma mtime


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
    """fn() cacheado mientras la firma (mtimes) no cambie.

    La firma se escribe solo si fn() tiene exito, y ya con el valor listo
    (update atomico): ni un fallo envenena la cache ni un lector
    concurrente ve la firma sin el valor.
    """
    if cache.get("sig") != sig:
        valor = fn()
        cache.update(sig=sig, valor=valor)
    return cache["valor"]


def _agregados() -> dict:
    """Analisis del parquet en memoria: universo y calidad.

    Lee solo las columnas del analisis (no los 600 MB enteros) y cachea por
    (mtime, tamano): releer solo cuando update.py reescribe el almecen.
    """
    p = DATA_DIR / "licitaciones.parquet"
    st = p.stat()

    def calc() -> dict:
        import pyarrow.parquet as pq

        n_cols = pq.ParquetFile(p).metadata.num_columns
        df = pd.read_parquet(p, columns=["conjunto", "ml_estado", "version",
                                         "score_calidad"])
        conjuntos = {}
        for c in CONJUNTOS:
            g = df[df["conjunto"] == c]
            est = g["ml_estado"]
            # pre-2021: fuera de la ventana de modelado, escrito por el
            # cleaning de 2026-09-12 — el dato vive en el parquet aunque el
            # codigo que lo genero se revirtiera; sin contarla, el universo
            # no cuadra con las filas del conjunto
            conjuntos[c] = {
                "train": int((est == "train").sum()),
                "test": int((est == "test").sum()),
                "filtered": int((est == "filtered").sum()),
                "pre_2021": int((est == "pre-2021").sum()),
                "abiertas": int(est.isna().sum()),
                "servidas": int(g["version"].notna().sum()),
                "calidad_media": round(float(g["score_calidad"].mean()), 3),
            }
        return {
            "filas": len(df), "columnas": n_cols,
            "mb": round(st.st_size / 1e6, 1),
            "modificado": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "conjuntos": conjuntos,
            "versiones_pred": {f"v{k}": int(v) for k, v
                               in df["version"].value_counts().items()},
        }

    return _cache((st.st_mtime_ns, st.st_size), _AGG_CACHE, calc)


# --- EDA del parquet (panel Exploratorio de Datos) -----------------------------
# Columnas del analisis EDA: identidad, objetivos y las dimensiones graficables.
COLS_EDA = ["conjunto", "fecha_publicacion", "cpv_principal", "tipo_contrato",
            "num_ofertas", "importe_con_iva", "importe_adj_con_iva",
            "importe_sin_iva", "importe_adjudicacion"]
DIMS_EDA = ("ano", "cpv", "tipo")


def _r2(v) -> float | None:
    """round(x, 2) con guardas NaN/vacio (NaN no es JSON)."""
    return None if v is None or pd.isna(v) else round(float(v), 2)


def _pct(n: int, total: int) -> float | None:
    return round(100 * n / total, 1) if total else None


def _serie_dim(cat: pd.Series, disc: pd.Series, num: pd.Series,
               orden=None) -> dict:
    """Agregados de una dimension: n, descuento medio y ofertas media por etiqueta.

    orden=None ordena por n descendente (identidad en cpv/tipo); una key de
    sort_index ordena por etiqueta (ano: cronologico con <2021 primero).
    """
    n = cat.value_counts()
    if orden is not None:
        n = n.reindex(sorted(n.index, key=orden))
    desc = disc.groupby(cat).mean().reindex(n.index)
    ofertas = num.groupby(cat).mean().reindex(n.index)
    return {
        "labels": n.index.astype(str).tolist(),
        "n": n.astype(int).tolist(),
        "desc": [_r2(v) for v in desc],
        "ofertas": [_r2(v) for v in ofertas],
    }


def _eda(conjunto: str) -> dict:
    """Stats basicas + agregados por dimension, cacheados por (mtime, conjunto)."""
    p = DATA_DIR / "licitaciones.parquet"
    st = p.stat()

    def calc() -> dict:
        df = pd.read_parquet(p, columns=COLS_EDA)
        if conjunto != "ambos":
            df = df[df["conjunto"] == conjunto]
        # descuento derivable con el criterio de cleaning (con-IVA en
        # licitaciones, sin-IVA en menores; negativos fuera, >70 truncados)
        # y num_ofertas valido — las medias van sobre poblacion evaluable
        es_lic = df["conjunto"] == "licitaciones"
        base = df["importe_con_iva"].where(es_lic, df["importe_sin_iva"])
        adj = df["importe_adj_con_iva"].where(es_lic, df["importe_adjudicacion"])
        disc = (1 - adj / base) * 100
        disc = disc.where(base.notna() & (base > 0) & adj.notna()
                          & (disc >= 0)).clip(upper=cleaning.DISCOUNT_RANGE[1])
        num = df["num_ofertas"].where(
            df["num_ofertas"].between(*cleaning.NUM_OFERTAS_RANGE))
        n_disc, n_num = int(disc.notna().sum()), int(num.notna().sum())

        # ano de publicacion, historial completo (cada ano su barra, tambien
        # los pre-2021), con la misma reparacion que cleaning.fix_ano (bug
        # del feed de anos a dos digitos; fuera de 2000..hoy = s/d);
        # cpv por division (2 primeros digitos)
        ano = cleaning.fix_ano(df["fecha_publicacion"].dt.year)
        cat_ano = ano.astype("Int64").astype(str).where(ano.notna())
        cat_cpv = (df["cpv_principal"].astype("string").str.slice(0, 2)
                   .fillna("s/d"))
        cat_tipo = df["tipo_contrato"].fillna("s/d")

        return {
            "filtro": {"conjunto": conjunto},
            "modificado": datetime.fromtimestamp(st.st_mtime).isoformat(),
            # firma de version de las figuras: cambia con el parquet (mtime)
            # O con el codigo que agrega/pinta (api.py, graficas.py) — la URL
            # del <img> la lleva para que el navegador no sirva un SVG viejo
            # cacheado como inmutable tras tocar el render
            "v": "-".join(str(x) for x in (
                st.st_mtime_ns, Path(__file__).stat().st_mtime_ns,
                (ROOT / "Dashboard" / "graficas.py").stat().st_mtime_ns)),
            "stats": {
                "filas": len(df),
                "descuento_medio": _r2(disc.mean()), "n_disc": n_disc,
                "pct_ceros": _pct(int((disc == 0).sum()), n_disc),
                "ofertas_media": _r2(num.mean()), "n_ofertas": n_num,
                "pct_sin_ofertas": _pct(int((num == 0).sum()), n_num),
            },
            "dims": {
                "ano": _serie_dim(cat_ano, disc, num, orden=lambda ix: ix),
                "cpv": _serie_dim(cat_cpv, disc, num),
                "tipo": _serie_dim(cat_tipo, disc, num),
            },
        }

    return _cache((st.st_mtime_ns, st.st_size, conjunto),
                  _EDA_CACHE.setdefault(conjunto, {}), calc)


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
                                      "learning_rate", "n_estimators", "device",
                                      "random_state", "n_jobs")},
        "rounds": meta["rounds"], "val_window": meta["early_stop_val_window"],
        "sizes": meta["sizes"], "n_features": len(meta["features"]),
        "encoding": (meta.get("encoding_organo") or {}).get("tipo"),
        "tam_ubj_mb": round((MODELS_DIR / f"{stem}.ubj").stat().st_size / 1e6, 1),
    }


def _lab_stem(linea: str) -> str:
    """licitaciones/num_ofertas -> experimentos/licitaciones_num_ofertas"""
    return linea.replace("/", "_") + ".trainings.json"


def _enc_label(name: str) -> str:
    """Etiqueta corta del encoding del organo, derivada del name del experimento."""
    if "teorg_m100nat" in name or "teorg_m20nat" in name:
        return f"TE m={'100' if 'm100' in name else '20'} + nativo"
    if "teorg_m100" in name:
        return "TE m=100"
    if "teorg_m20" in name:
        return "TE m=20"
    if "freqorg" in name:
        return "frecuencia"
    return "nativo"


def _laboratorio() -> dict:
    """Campeones promovidos + resumen del registro del Lab, cacheado por mtime."""
    def calc() -> dict:
        campeones: dict[str, dict] = {}
        entradas, ts_todas = 0, []
        for linea, name in PROMOVIDOS_LAB.items():
            ruta = LAB_DIR / _lab_stem(linea)
            reg = json.loads(ruta.read_text())
            entradas += len(reg)
            ts_todas.extend(e["ts"] for e in reg)
            e = [x for x in reg if x["name"] == name][-1]  # ultima con ese name
            alg = e["spec"]["algoritmo"]
            p = alg["params"]
            metrica = "auc" if "auc" in e["metricas"]["test"] else "mae"
            # mejor baseline tonto en test (mae en regresion, logloss en zd)
            bk = "logloss" if metrica == "auc" else "mae"
            base = min((b[bk] for b in e["baselines"]["test"].values()
                        if b.get(bk) is not None), default=None)
            c, l = campeones.setdefault(linea.split("/")[0], {}), linea.split("/")[1]
            c[l] = {
                "name": name, "ts": e["ts"], "device": e["huella"]["device"],
                "best_it": alg["best_iteration"],
                "n_features": len(e["spec"]["features"]),
                "encoding": _enc_label(name),
                "hp": {k: p[k] for k in ("objective", "max_depth", "eta", "eval_metric")},
                "val": e["metricas"]["val"], "test": e["metricas"]["test"],
                "metrica": metrica, "mejor_baseline_test": base,
                "n_corridas": len(reg),
            }
        return {"disponible": True,
                "registro": {"entradas": entradas,
                             "ultima": max(ts_todas) if ts_todas else None},
                "campeones": campeones}

    if not LAB_DIR.is_dir():
        return {"disponible": False, "registro": None, "campeones": {}}
    sig = tuple((LAB_DIR / _lab_stem(l)).stat().st_mtime_ns for l in PROMOVIDOS_LAB)
    return _cache(sig, _LAB_CACHE, calc)


# ---------------------------------------------------------------------------
# Ops en subprocess (single-flight): un op = 1..N pasos encadenados
# ---------------------------------------------------------------------------
_OPS_LOCK = threading.Lock()
_OPS_STATE: dict = {"op": None, "run_id": None, "started": None, "rc": None,
                    "log": None, "log_run_id": None, "paso": None, "pasos": []}


def _pasos_de(op: str, modo: str, meses: int) -> list[tuple[str, list[str]]]:
    """(nombre_paso, cmd) por paso del op. Puro, para poder testearlo."""
    pasos: list[tuple[str, list[str]]] = []
    for nombre in (CICLO if op == "ciclo" else (op,)):
        flags: list[str] = []
        if nombre == "evaluar":
            # en el ciclo siempre prepromote: cierra el modelo reemplazado
            flags += ["--modo", "prepromote" if op == "ciclo" else modo]
        if nombre == "update":
            flags += ["--meses", str(meses)]
        pasos.append((nombre, [sys.executable, "-u", OPS[nombre], *flags]))
    return pasos


def _start_op(op: str, pasos: list[tuple[str, list[str]]]) -> dict:
    """Lanza la cadena de pasos en un hilo watcher; no bloqueante."""
    with _OPS_LOCK:
        if _OPS_STATE["op"] is not None:
            raise HTTPException(409, f"ya hay un op en marcha: {_OPS_STATE['op']}")
        run_id = time.strftime("%Y%m%d_%H%M%S")
        _OPS_STATE.update(op=op, run_id=run_id, started=_ahora(), rc=None,
                          log=None, log_run_id=None, paso=None,
                          pasos=[{"paso": n, "status": "pendiente"} for n, _ in pasos])

    def _watch() -> None:
        rc = 0
        for nombre, cmd in pasos:
            # paso suelto: el run_id del op; paso de ciclo: el suyo propio
            # (los agrupa el campo "ciclo" en runs.jsonl)
            paso_rid = run_id if len(pasos) == 1 else time.strftime("%Y%m%d_%H%M%S")
            log = OPS_DIR / "logs" / f"{nombre}_{paso_rid}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with _OPS_LOCK:
                _OPS_STATE.update(paso=nombre, log=str(log), log_run_id=paso_rid)
                for p in _OPS_STATE["pasos"]:
                    if p["paso"] == nombre:
                        p["status"] = "en_marcha"
            started = _ahora()
            with open(log, "ab") as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
            rec = {"op": nombre, "run_id": paso_rid, "triggered": "api",
                   "started": started, "finished": _ahora(),
                   "status": "ok" if rc == 0 else "failed", "rc": rc}
            if len(pasos) > 1:
                rec["ciclo"] = run_id
            with open(OPS_DIR / "runs.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with _OPS_LOCK:
                for p in _OPS_STATE["pasos"]:
                    if p["paso"] == nombre:
                        p["status"] = rec["status"]
            if rc != 0:
                break
        with _OPS_LOCK:
            _OPS_STATE.update(op=None, rc=rc, paso=None)

    threading.Thread(target=_watch, daemon=True).start()
    if len(pasos) == 1:
        log = OPS_DIR / "logs" / f"{op}_{run_id}.log"
        return {"op": op, "run_id": run_id, "log": str(log),
                "cmd": " ".join(pasos[0][1])}
    return {"op": op, "run_id": run_id, "pasos": [n for n, _ in pasos]}


def _ops_estado() -> dict:
    with _OPS_LOCK:
        e = dict(_OPS_STATE)
        e["pasos"] = [dict(p) for p in _OPS_STATE["pasos"]]
        return e


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


@app.get("/eda")
def eda(conjunto: str = "ambos") -> dict:
    """Stats basicas + agregados por dimension del parquet (panel EDA de Datos).

    ?conjunto=ambos|licitaciones|menores. Filas cuenta todo el filtro; las
    medias (% descuento, nº ofertas) van solo sobre poblacion evaluable —
    descuento derivable en [0, 70] (criterio de cleaning) y num_ofertas en
    [0, 50]. Cacheado por (mtime, conjunto): cambiar filtro no relee el
    parquet si ya esta computado.
    """
    if conjunto not in ("ambos", *CONJUNTOS):
        raise HTTPException(400, "conjunto debe ser ambos|licitaciones|menores")
    return _eda(conjunto)


@app.get("/eda/fig/{dim}")
def eda_fig(dim: str, conjunto: str = "ambos", tema: str = "claro") -> Response:
    """La figura de una dimension como SVG matplotlib (Dashboard/graficas.py).

    Barras = nº licitaciones por etiqueta; lineas = descuento medio (%) y nº
    ofertas medio. El dato sale del cache de /eda y el SVG se cachea por
    (dim, conjunto, tema) con firma de mtime — la URL del <img> versiona con
    &v=<modificado>, asi el navegador trata cada version como inmutable.
    """
    if dim not in DIMS_EDA:
        raise HTTPException(400, f"dimension desconocida: {dim} "
                                 f"(validas: {'|'.join(DIMS_EDA)})")
    if conjunto not in ("ambos", *CONJUNTOS):
        raise HTTPException(400, "conjunto debe ser ambos|licitaciones|menores")
    if tema not in graficas.TEMAS:
        raise HTTPException(400, "tema debe ser claro|oscuro")
    d = _eda(conjunto)["dims"][dim]
    st = (DATA_DIR / "licitaciones.parquet").stat()
    cache = _EDA_FIG_CACHE.setdefault((dim, conjunto, tema), {})
    svg = _cache((st.st_mtime_ns, st.st_size), cache,
                 lambda: graficas.fig_eda(d["labels"], d["n"], d["desc"],
                                           d["ofertas"], tema, conjunto))
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=604800, immutable"})


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


@app.get("/laboratorio")
def laboratorio() -> dict:
    """Los campeones del Lab promovidos a este motor + su registro.

    La fuente es experimentos/ de ../Licitaciones-Lab (append-only, el
    BUILDLOG del Lab decide los campeones): PROMOVIDOS_LAB fija que experimento
    se promociono en cada linea — el puerto a mano actualiza la constante.
    """
    return _laboratorio()


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


@app.post("/ops/{op}")
def lanzar_op(op: str, modo: str = "curso", meses: int = 12):
    """Lanza un op en subprocess (single-flight): evaluar | update | cleaning |
    featurer | training | ciclo.

    ciclo encadena el protocolo completo (update -> evaluar prepromote ->
    cleaning -> featurer -> training) y para al primer paso fallido; cada paso
    deja su linea en runs.jsonl y su log. Query: modo (solo evaluar suelto:
    curso|prepromote) · meses (update y ciclo).
    """
    if op not in OPS and op != "ciclo":
        raise HTTPException(404, f"op desconocido: {op} "
                                 f"(disponibles: {', '.join((*OPS, 'ciclo'))})")
    if modo not in ("curso", "prepromote"):
        raise HTTPException(400, "modo debe ser curso|prepromote")
    if meses < 1:
        raise HTTPException(400, "meses debe ser >= 1")
    return _start_op(op, _pasos_de(op, modo, meses))


@app.get("/ops/log/{op}/{run_id}")
def ver_log(op: str, run_id: str, cola: int = 120):
    """Ultimas lineas del log de un run — el visor del dashboard.

    El nombre del log es {op}_{run_id}.log (exacto, sin glob: dos pasos de
    un ciclo pueden caer en el mismo segundo y compartir run_id).
    """
    if not re.fullmatch(r"[a-z_]+", op):
        raise HTTPException(400, "op invalida")
    if not re.fullmatch(r"\d{8}_\d{6}", run_id):
        raise HTTPException(400, "run_id invalido")
    log = OPS_DIR / "logs" / f"{op}_{run_id}.log"
    if not log.exists():
        raise HTTPException(404, f"sin log para {op} {run_id}")
    return {"op": op, "run_id": run_id, "log": str(log),
            "lineas": log.read_text(errors="replace").splitlines()[-cola:]}


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
        res = inference.inferir(df, MODELS_DIR)
    except ValueError as e:
        raise HTTPException(409, str(e))
    buf = io.BytesIO()
    res.to_parquet(buf, index=False, compression="snappy")
    nombre = Path(file.filename or "subida.parquet").stem + "_preds.parquet"
    return Response(content=buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition":
                             f'attachment; filename="{nombre}"',
                             "X-Modelo-Version": res["version"].iloc[0],
                             "X-Filas": str(len(res))})


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(DASHBOARD.read_text())
