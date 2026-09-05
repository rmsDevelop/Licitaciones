# Licitaciones

La contratación pública española (licitaciones + contratos menores,
sindicación PLACSP) como un pipeline de datos + ML en pocos archivos:
scrape → limpieza → features → 6 líneas xgboost → servido a demanda,
todo operado desde una API localhost con dashboard.

## Árbol

- `Scraper/scrape.py` — carga completa PLACSP → `Data/licitaciones.parquet`
  (41 raw + 20 calidad + score + esquema de inferencia vacío).
- `Scraper/update.py` — refresco incremental de los últimos N meses;
  marca `test` a la adjudicación recién llegada (protocolo del ciclo).
- `Modeling/` — `cleaning.py` (clasifica `ml_estado`) · `featurer.py`
  (`Data/features.parquet`: 19 features `f_*`) · `training.py` (6 líneas
  xgboost → `Models/`).
- `Inference/` — `inference.py` (servido a demanda + router, sin
  dependencia de `Data/`) · `evaluate.py` (evaluación del modelo expuesto
  sobre las filas test).
- `api.py` + `Dashboard/` — la consola de operación: todo el pipeline
  corre como op subprocess single-flight, incluido el ciclo completo.

## Arranque

```bash
# venv (una vez)
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

# consola de operación
.venv/bin/uvicorn api:app --port 8000   # dashboard → http://127.0.0.1:8000/
```

`Data/` (~9 GB) y `Models/` son artefactos gitignored, regenerables con
el pipeline. La API es localhost por diseño: exposición, auth y TLS
pertenecen al sistema que se ponga delante.

## API

Consulta (la alimentan `Data/`, `Models/` y el Lab hermano):

- `GET /status` — universo por conjunto (train/test/filtered/abiertas),
  versión expuesta, op en marcha.
- `GET /datos` — KPIs y desgloses del parquet: universo, filas por año
  de publicación, calidad media, preds por versión.
- `GET /modelos` — metas de las 6 líneas (receta, rounds, features) +
  umbrales del router; nunca los boosters.
- `GET /laboratorio` — campeones promovidos desde `../Licitaciones-Lab`;
  sin el repo hermano al lado responde `disponible: false`.
- `GET /evaluaciones` — evaluación en curso (`estado_curso.json`) +
  histórico de cierres prepromote (`historico.jsonl`).

Operación del pipeline (subprocess single-flight; logs en `Data/ops/`,
una línea por run en `Data/ops/runs.jsonl`):

- `POST /ops/ciclo?meses=12` — el protocolo de reentrenamiento completo:
  update (marca test) → evaluar prepromote (cierra el modelo saliente) →
  cleaning (pliega test a train) → featurer → training.
- `POST /ops/{evaluar|update|cleaning|featurer|training}` — pasos
  sueltos (`?modo=curso|prepromote` en evaluar, `?meses=N` en update);
  scrape queda fuera (reconstruir el almacén entero es cosa de terminal).
- `GET /ops/log/{op}/{run_id}` — últimas líneas del log de un run.

Servido (el producto):

- `POST /inferir` — multipart con el parquet raw (columna `conjunto`,
  tope 256 MB) → el mismo parquet con las 5 preds + `version`; headers
  `X-Modelo-Version` y `X-Filas`.

```bash
# ejemplos
curl -X POST 'http://127.0.0.1:8000/ops/ciclo?meses=12'
curl -F 'file=@entrada.parquet' http://127.0.0.1:8000/inferir -o salida.parquet
```

## Dashboard

Cuatro paneles con deep-link (`#lab #historico #expuesto`):

1. **Datos** — universo del parquet, filas por año y ciclo de vida,
   calidad media, preds por versión.
2. **Laboratorio** — los campeones del Lab promovidos a producción.
3. **Histórico** — cierres prepromote de los modelos ya reemplazados y
   tendencia del system MAE entre versiones.
4. **Modelo expuesto** — versión, edad, umbrales del router, evaluación
   en curso, lanzamiento del ciclo o pasos sueltos y visor de logs.
