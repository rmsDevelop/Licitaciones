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
- El mensual del mes en curso aún no existe al principio de mes: PLACSP
  devuelve un placeholder (se tolera). Ojo: el cache lo trata como
  válido — scrape.py sin `--force` no lo refresca; `update.py` fuerza su
  ventana y lo auto-cura.

### Duración del run completo (referencia)

Descarga ~8 GB ≈ 10 min (línea rápida, cacheable); parse licitaciones
≈ 20 min; descarga+parse menores ≈ 45 min; calidad ≈ 5 min. Total < 1,5 h.

---

## 2026-09-02 — `Scraper/update.py`

**Estado: HECHO.** Primer run real (`--meses 12`) contra `Data/` el
2026-09-02: **3.630.925 filas × 70 columnas (628 MB)** — licitaciones
867.893 (**212 nuevos**, 2.169 cambiados de 217.968 refrescados) +
menores 2.763.032 (178 nuevos, 17 cambiados). Duración ~17 min
(~2,3 GB re-descargados). Post-run verificado: 0 ids duplicados,
esquema == `FINAL_SCHEMA`, nuevas con inferencia vacía. Los "nuevos" no
son solo lo publicado hoy: el primer lote del ZIP de septiembre
(publicado ese día, 10 MB con fechas 21-ago..1-sep) arrastra registros de
última hora de agosto cuya primera aparición en la sindicación es ese
lote.

Qué hace: refresco incremental de `licitaciones.parquet` con los últimos
X meses (`--meses`, default 12, alineado a mes natural: mes en curso +
X-1 anteriores), reutilizando las primitivas de `scrape.py` (descarga
atómica, parse streaming, calidad, esquema, escritura atómica).

Pipeline: ventana → re-descarga **forzada** de los ZIPs cuyo periodo
interseca (mensuales `YYYYMM >=` inicio; si la ventana nace antes de 2025,
el anual del año de inicio) → parse+calidad solo de las filas de la
ventana → merge contra el parquet existente → escritura atómica con el
mismo `FINAL_SCHEMA`.

Merge (3 reglas, portadas del sistema anterior `spts/inference/store.py`):
id nuevo → fila completa con inferencia vacía; id conocido → raw (41) +
calidad (22) frescos, `preds`/`ml_estado`/`version` se conservan; id
ausente de la ventana → fila intacta. No clasifica (`Modeling/cleaning.py`)
ni re-infiere (`Inference/`): las preds de una fila refrescada pueden
quedar desfasadas hasta la próxima inferencia.

### Decisiones consensuadas

- Calidad de la ventana con cuantiles de la ventana: INT-FIA-01/09 (los
  únicos dataset-relativos) se calculan contra los meses de la ventana, no
  contra el histórico. Verificado con el run: drift de `score_calidad`
  medio −0,02, solo 1.218/126.576 filas de ventana cambian de score.
- Reutilización de descarga vía parámetro opcional `archivos=None` en
  `descargar_conjunto` (scrape.py, backwards-compatible) en vez de
  duplicar el loop.
- Ventana por ZIP (periodo), no por `fecha_publicacion`: los ZIPs
  contienen filas publicadas antes con actividad reciente — es como
  entran las adjudicaciones tardías.
- El placeholder del mes en curso (ZIP de ~0 MB) se tolera; como update
  fuerza siempre su ventana, el cache se auto-cura el próximo run (con
  scrape.py sin `--force` ese ZIP quedaría congelado hasta un force).

### Verificación (smoke `--meses 3`, sandbox con parquet copiado)

- Salida: 3.630.535 filas × 70 columnas, 0 ids duplicados, esquema ==
  `FINAL_SCHEMA` (tipos incluidos). Recuentos por conjunto exactos.
- Camino "ids nuevos" ejercitado quitando 1.000 ids de la ventana de la
  tabla vieja: volvieron como nuevos, inferencia vacía, calidad presente,
  recuento restaurado, sin duplicados.
- Filas fuera de la ventana: intactas bit a bit en score (3.503.959).
- `cambiados` compara `fecha_updated` (atom:updated) viejo vs nuevo: con
  el scrape de ayer, 0 cambios (contenido idéntico un día después).

### Operativa

- Ejecutar: `.venv/bin/python -u Scraper/update.py [--meses 12] [--data-dir DIR]`
- Coste ~17 min con `--meses 12` (~2,3 GB de mensuales re-descargados,
  cache reutilizable). Requiere `licitaciones.parquet` previo.

---

## Pendiente (orden propuesto, a consensuar)

1. `Modeling/cleaning.py` — clasificar training/filtered (criterios de
   Licitaciones-Lab; aquí entra el recorte de población ≥2021 si procede).
2. `Modeling/featurer.py` → `Modeling/training.py` → `Inference/` → `Dashboard/`.

`requirements.txt` + venv propio + git (main, `Data/` ignorado) quedaron
listos el 2026-09-02, antes de iniciar update.py.
