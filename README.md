# Nuevo_Licitaciones

Rehacer de `../Licitaciones` sin overengineering: la contratación pública
española (licitaciones + contratos menores, sindicación PLACSP) como un
pipeline de datos + ML en pocos archivos. El registro de construcción
sesión a sesión — decisiones, bugs y verificaciones — vive en
[BUILDLOG.md](BUILDLOG.md): **esa es la documentación del proyecto**.

## Árbol

- `Scraper/scrape.py` — carga completa PLACSP → `Data/licitaciones.parquet`
  (41 raw + 20 calidad + score + esquema de inferencia vacío).
- `Scraper/update.py` — refresco incremental de los últimos N meses;
  marca `test` a la adjudicación recién llegada (protocolo del ciclo).
- `Modeling/` — `cleaning.py` (clasifica `ml_estado`) · `featurer.py`
  (`Data/features.parquet`, unión de features del Lab) · `training.py`
  (6 líneas xgboost → `Models/`).
- `Inference/` — `inference.py` (servido a demanda + router) ·
  `evaluate.py` (evaluación del expuesto sobre las filas test).
- `api.py` + `Dashboard/` — API localhost de operación: todo el pipeline
  corre como op subprocess single-flight, incluido el ciclo completo.

## Operativa

```bash
# venv (una vez)
uv venv .venv --python 3.12 && uv pip install -r requirements.txt

# consola de operación
.venv/bin/uvicorn api:app --port 8000   # → http://127.0.0.1:8000/
```

- **Ciclo de reentrenamiento** — botón «Lanzar ciclo» del dashboard o
  `curl -X POST 'http://127.0.0.1:8000/ops/ciclo?meses=12'`:
  update (marca test) → evaluar prepromote (cierra el modelo saliente) →
  cleaning (pliega test a train) → featurer → training.
- **Ops sueltos** — `POST /ops/{evaluar|update|cleaning|featurer|training}`;
  logs en `Data/ops/logs/`, una línea por paso en `Data/ops/runs.jsonl`.
- **Servido (el producto)** — `POST /inferir`: parquet raw con columna
  `conjunto` → el mismo parquet con las 5 preds + versión.

`Data/` (~9 GB) y `Models/` son artefactos gitignored, regenerables con
el pipeline. La API es localhost por diseño: exposición, auth y TLS
pertenecen al sistema que se ponga delante.
