# BUILDLOG — Nuevo_Licitaciones

Registro de construcción, sesión a sesión. Objetivo del repo: rehacer
`../Licitaciones` sin overengineering, archivo a archivo y consensuando
cada paso. El árbol de ficheros lo define el usuario; nada se crea sin
permiso explícito.

---

## 2026-09-01/02 — `Scraper/scrape.py`

**Estado: HECHO.** `Data/licitaciones.parquet` generado y verificado:
**3.630.535 filas × 70 columnas (600 MB)** — licitaciones 867.681 +
menores 2.762.854, 0 ids duplicados.

Qué hace: descarga los ZIPs PLACSP de `licitaciones` y `menores`
(anuales hasta 2024, mensuales desde 2025; cache en
`Data/downloads/placsp/`), parsea el ATOM a las 41 columnas raw del
esquema congelado del sistema anterior, aplica los 20 indicadores de
calidad (BORME/TED con descarga automática del release de GitHub de
licitaciones-espana si faltan), crea las columnas de inferencia vacías
y escribe el parquet final de forma atómica. Código portado del vendor
probado de `Licitaciones`, con sus fixes.

### Decisiones consensuadas

- Un solo parquet con ambos conjuntos (columna `conjunto`), no uno por conjunto.
- Predicciones **persistidas por fila**: 5 columnas pred + `ml_estado` +
  `version`, creadas vacías aquí y rellenadas por fases posteriores.
  `ml_estado`: null (abierta) | test | train | filtered.
- Los 20 indicadores de calidad, con referencias BORME/TED de descarga
  automática; si no se consiguen, la columna existe a NaN (esquema estable).
- Todo en `scrape.py` (un fichero), sin tests, comentarios en español.
- Ventana por defecto `2021-<año actual>`; `--force` re-descarga todo
  (PLACSP actualiza los ZIPs in place: las adjudicaciones tardías solo
  entran así).
- Sin recorte por `fecha_publicacion`: los ZIPs 2021+ contienen ~29k
  filas publicadas antes (89% `Resuelta` — actividad del expediente
  dentro de la ventana; ver análisis en el historial de la sesión).
  El recorte de población es decisión de `Modeling/cleaning.py`.
- `version` (no `ml_train_version` como en el sistema viejo).

### Bugs encontrados con el run real (y corregidos)

1. **Descargas no atómicas**: interrumpir el proceso dejaba un ZIP
   truncado que el cache (>1000 bytes) trataba como válido — borró del
   run el año 2023 entero hasta detectarlo. Fix: `.part` + `replace`.
2. **Extracción a `/tmp`** (tmpfs de 6 GB en WSL2): los ZIPs anuales
   descomprimen varios GB → "No space left on device". Fix: parse en
   streaming desde el ZIP (`zf.open` → `ET.iterparse`), sin extraer.
3. **RAM**: acumular dicts/DataFrames de los 25 ZIPs no cabe (menores =
   2.75M filas; 11 GB de máquina). Fix: tabla Arrow por ZIP, dedupe
   global keep-last sobre el concatenado, pandas solo un conjunto cada
   vez, calidad asignando columnas in-place (sin copia del DataFrame).
4. **Esquemas Arrow explícitos** (`RAW_SCHEMA`/`FINAL_SCHEMA`): la
   inferencia produce `null()` cuando un ZIP trae una columna entera a
   null (p.ej. `es_pyme`) y rompe la concatenación.
5. **Colisión de nombres**: la columna de ciclo de vida se llamaba
   `estado` y sobrescribía el estado raw del expediente → `ml_estado`.
6. `score_calidad` tipado bool en el esquema final (es float64) —
   cazado validando la cadena con un ZIP real antes del run completo.
7. DeprecationWarnings de truthiness de `Element` → `is not None`.

### Esquema de salida (70 columnas, `FINAL_SCHEMA` en código)

41 raw (dtypes del esquema congelado: fechas `timestamp[us]`,
`fecha_updated` `timestamp[us, UTC]`, importes y `num_ofertas` float64,
`es_pyme` bool) + 20 `INT-*` (bool) + `score_calidad` (float64) +
`es_menor` (bool) + 5 preds (float32×4, int8) + `ml_estado`/`version`
(string). Contrato: `python -c "import sys; sys.path.insert(0,'Scraper'); import scrape; print(scrape.FINAL_SCHEMA)"`.

### Operativa

- Ejecutar: `.venv/bin/python -u Scraper/scrape.py [--anos 2021-2026] [--force] [--data-dir DIR]`
  (`-u`: sin él, el stdout buffereado oculta el progreso).
- **Python: venv propio** (`.venv`, CPython 3.12.14 via uv) con
  `requirements.txt` pinado al runtime probado. Recrear:
  `uv venv .venv --python 3.12 && uv pip install -r requirements.txt`.
  Las dependencias de fases posteriores (xgboost, fastapi…) se añaden
  cuando existan esos ficheros.
- **Git**: repo en rama `main`. `Data/` (ZIPs, referencias, parquet,
  ~9 GB) y `.venv/` están gitignored — solo entra código y documentación.
- En disco: `Data/downloads/` ~8 GB de ZIPs (cache, re-utilizable),
  `Data/references/` ~490 MB, parquet 600 MB. ~9 GB en total.
- El mensual del mes en curso aún no existe: PLACSP devuelve un
  placeholder (se tolera; se re-descarga el próximo run).

### Duración del run completo (referencia)

Descarga ~8 GB ≈ 10 min (línea rápida, cacheable); parse licitaciones
≈ 20 min; descarga+parse menores ≈ 45 min; calidad ≈ 5 min. Total < 1,5 h.

---

## Pendiente (orden propuesto, a consensuar)

1. `Scraper/update.py` — refresco incremental de los últimos X meses
   (reutiliza las primitivas de scrape.py; preserva `ml_estado`/preds).
2. `Modeling/cleaning.py` — clasificar training/filtered (criterios de
   Licitaciones-Lab; aquí entra el recorte de población ≥2021 si procede).
3. `Modeling/featurer.py` → `Modeling/training.py` → `Inference/` → `Dashboard/`.

`requirements.txt` + venv propio + git (main, `Data/` ignorado) quedaron
listos el 2026-09-02, antes de iniciar update.py.
