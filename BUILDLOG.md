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

## 2026-09-02 — `Modeling/` (cleaning.py + featurer.py + training.py)

**Estado: HECHO.** Pipeline ML completo portado de Licitaciones-Lab y
ejecutado sobre `Data/licitaciones.parquet` (3.630.925 filas):
`cleaning.py` clasifica `ml_estado`, `featurer.py` genera
`Data/features.parquet` (3.516.651 filas train × 68 cols) y `training.py`
entrena las 6 líneas y guarda `Models/` (6 `.ubj` + 6 `.meta.json`).

### cleaning.py

Rellena `ml_estado` in place (escritura atómica, mismo esquema/compresión que
scrape.py): **'train' | 'filtered' | null (abierta)**. Se ejecuta tras la
evaluación: toda fila válida pasa a 'train' (sin reserva de test); el split
train/val lo hace training.py. Recalcula desde las columnas raw en cada run
(idempotente, sin historial). Precedencia: fuera de ventana → 'filtered';
expediente abierto → null; cerrado en ventana → 'train' si algún objetivo es
válido ('filtered' si no). Criterios del Lab: `fix_ano` (bug 2 dígitos, sanidad
2000..año en curso), ventana ≥2021; licitaciones exige `estado ∈
{Resuelta, Adjudicada}`; válida = num_ofertas ∈ [0,50] **o** importes sanos
con discount_pct ∈ [0,70] (licitaciones base con-IVA, menores sin-IVA sin
filtro de estado). Resultado: lic 756.555 train / 42.233 filtered / 69.105
abiertas; menores 2.760.096 / 1.457 / 1.479. Verificado en sandbox:
idempotente, 0 train fuera de ventana o sin objetivo, 0 abiertas con datos,
0 ids duplicados, esquema intacto.

### featurer.py

`Data/features.parquet` = filas train de ambos conjuntos: `id/conjunto/ano/
fecha_publicacion` (claves) + 3 objetivos (`num_ofertas`, `discount_pct`
derivado por base del conjunto, `zero_discount`=(disc==0)) + **unión** de
features de las 6 líneas del Lab (mismas transformaciones: CPV/NUTS/dinero/
duración/tiempo/keywords de objeto/missingness/volúmenes prior-año; mismas
reglas de leakage). Donde las líneas del Lab discrepaban, la tabla lleva
variantes con sufijo (`budget_to_estimado_con_iva_raw|_cap`,
`log_duracion_days_raw`, `log_budget_sin_iva`, `importe_sin_iva` kept en
menores) y **`FEATURES_LINEA`** (dict en el código) define la lista exacta por
línea — el contrato compartido con training.py e Inference/. Los INTs de
validez (VAL-01/04/05/14, FIA-11) caen por varianza cero en población limpia
(el Lab los dropeaba igual). `row_missing_count` sobre raw+calidad+objetivos.

### training.py

**Elabora los 6 modelos — nada más.** No evalúa (no hay test split: la
evaluación ocurrirá en otra fase) ni enruta (el gate zero_discount→0 lo
decide quien sirva). 6 entrenamientos secuenciales con GPU (GTX 1660 Ti,
`device=cuda`, fallback CPU por línea). HPs = campeones del Lab: lic num
**log1p+TE d13 lr.03** | lic zd **d14 lr.05 spw=auto** | lic disc
**pseudohuber d14 lr.03 sub.9** | men num **log1p+TE d13 lr.03** | men zd
**d13 lr.03 spw=none** | men disc **pseudohuber+TE d14 lr.03**. TE =
encodings suavizados (α=30) de órgano/dir3/ciudad fit en TRAIN, mapas
guardados en el meta. Receta: población por objetivo → split cronológico
(VAL = últimos 6 meses, **solo como ventana de early-stopping**) → sonda
early-stop TRAIN vs VAL → rounds congelados (+buffer 50) → **booster de
producción** = refit con todas las filas y rounds congelados → `Models/
<linea>.ubj` + `.meta.json`. Las seis líneas son independientes entre sí.

### Verificación de la sesión

- cleaning: invariantes en sandbox (idempotencia, train⊆ventana, 0 train sin
  objetivo válido, 0 abiertas con datos, esquema intacto, 0 ids duplicados).
- featurer: sin columnas de fuga, spot-checks de valores contra el raw,
  contrato FEATURES_LINEA validado al vuelo.
- training: smoke con muestra al 5% (6 líneas, cadena completa) + run real
  íntegro; test del contrato de servido cargando los 6 `.ubj` con sus metas
  (TE por mapas, niveles congelados, transform/clip) — predicciones sanas.
- Durante la sesión se midieron además, como comprobación puntual (código ya
  retirado), métricas de VAL con la sonda: lic num MAE 1,08 · lic zd AUC-PR
  0,911 · lic disc 8,38 · men num 0,29 · men zd 0,994 · men disc 1,35; y el
  sistema con gate (umbral elegido en VAL): lic 8,21 (Lab 6,78) · men 1,31
  (Lab 1,42). Referencias del Lab sobre ventanas TEST distintas — solo
  orientativas. La evaluación formal y el enrutado pertenecen a otra fase.

### Contrato de Inference/ (verificado cargando los .ubj + meta)

- Features: `meta['features']`; para columnas `_te` construir desde la raw
  (claves de `meta['target_encoding']['maps']`) con `.map(maps).fillna(gmean)`.
- Categóricas: `pd.Categorical` con los niveles de `meta['categorical_levels']`;
  niveles no vistos → enmascarar a NA antes (`s.where(s.isin(niveles))` — el
  constructor directo con valores fuera está deprecado en pandas y cambiará a
  error); el resto de columnas a float32.
- Salida: num_ofertas → `expm1` + clip [0,50]; zero_discount → prob (la
  comparación con el umbral y el gate del sistema los decide la fase de
  evaluación/enrutado, no hay umbral en el meta); discount → clip [0,70].
  Servir en cadena num → zd → discount (discount puede consumir
  `num_ofertas_pred` como feature serve-time).

### Notas operativas

- Ejecutar: `.venv/bin/python -u Modeling/{cleaning,featurer,training}.py`
  (`--data-dir`, `--dry-run` en cleaning; `--solo <substring>` y `--device` en
  training). Orden: update → (evaluación) → cleaning → featurer → training.
- `requirements.txt`: añadidos xgboost 3.4.1, scikit-learn 1.9.0, scipy 1.18.1.
- `Models/` gitignored (regenerable): ojo, los boosters son grandes (~5 GB
  totales; d13/d14 con >1000 rounds sobre millones de filas).
- `Data/features.parquet` (188 MB) también gitignored (regenerable).

---

## 2026-09-02 — `Inference/inference.py`

**Estado: HECHO.** Servido a demanda verificado contra el camino de
entrenamiento: **bit-exact** en filas con features idénticas.

Qué hace: recibe un conjunto de licitaciones (esquema raw+INT de scrape.py,
columna `conjunto`) y devuelve esas filas con las 5 preds + `version`
rellenas. Función importable `inferir(df)` (la usará api.py) + CLI
`--input archivo.parquet [--salida]`. No toca `ml_estado` (eso es de
cleaning.py) ni escribe en `licitaciones.parquet`: quien llama decide qué
filas manda. Receta: featurizar reutilizando `Modeling/featurer.py` (mismo
código, sin copia) → hist_volume_* de la tabla train (`features.parquet`)
reindexados → matriz por línea según el meta (TE por mapas, niveles
congelados, resto float32) → num `expm1`+clip[0,50] · zd prob · disc
clip[0,70] → router → version = max `meta['creado']` del set de 6.

### Decisiones consensuadas

- **A demanda**: no escanea el parquet buscando abiertas; el archivo recibe
  el conjunto y lo sirve entero (sin filtro de población).
- Función + CLI; la persistencia de preds en `licitaciones.parquet` es del
  llamador (api.py / flujo operativo).
- **Umbral del router** (la decisión "dónde vive" del pendiente): derivado
  con los boosters de producción sobre la ventana VAL de features.parquet y
  congelado como constante por conjunto en `inference.py` — lic 0,475
  (MAE sistema 6,14→5,96; precision/recall zero 0,90/0,93) · men 0,585
  (1,18→1,15; 0,98/0,98). Curva plana; algo optimista (VAL entra en el refit
  de producción) — la evaluación formal es de evaluate.py.
- hist_volume con base en la tabla train (réplica exacta del entrenamiento;
  requiere `Data/` accesible).
- row_missing_count de servido cuenta solo raw+calidad (objetivos
  excluidos): en abiertas serían +2/3 NaN sistemáticos en toda fila.
- version = max `meta['creado']`: identifica el modelo que predijo.

### Detalles de fidelidad (salieron en la verificación)

- `aplicar_hist` enmascara el bucket `"missing"` post-cast: en entrenamiento
  el recuento era pre-cast (segmento NaN → hist NaN); sin máscara esas filas
  recibirían el recuento de un bucket que no existía en train.
- `INT-VAL-14` está a NaN en toda la parte licitaciones de features.parquet
  (constante en población limpia → dropeada por conjunto; la union la
  rellena de NaN). Las líneas de lic la entrenaron **inerte** (nanificarla
  no cambia preds: verificado). En servido se alimenta el valor raw — lo
  correcto de cara a reentrenamientos.
- La métrica in-sample exige **filtrar** la población de descuento a [0,70],
  no clipear: 110/500 cerradas de licitaciones quedan fuera de rango y con
  clip inflaban el MAE de 6 a 11 (falso positivo mío, ya en el test).

### Verificación

- Estructura: dtypes float32×4 + int8 + version única; router consistente
  (zd_pred == prob ≥ umbral; system = gate); ids/filas/ml_estado intactos;
  preds sin NaN.
- Equivalencia con la tabla train en 1.000 cerradas: features idénticas
  salvo row_missing_count (±2, por diseño, filas con objetivos parcialmente
  NaN en train) e INT-VAL-14 (inerte en lic). En filas 100% idénticas
  (440 lic / 497 men) las preds son **bit-exact** (≤ 4e-7 = redondeo
  float32 del expm1).
- Determinismo bit a bit en repeticiones; MAE in-sample con población
  filtrada: system 6,19 lic · 0,90 men (VAL: 5,96 · 1,15).
- Smoke en `/tmp/smoke_inf/` (1.500 abiertas + 500 cerradas por conjunto);
  código de verificación retirado.

### Operativa

- Ejecutar: `.venv/bin/python -u Inference/inference.py --input X.parquet
  [--salida Y.parquet] [--models-dir Models] [--feats Data/features.parquet]`
- Coste: ~14 s para 4k filas (dominado por carga de boosters + base hist).
- Sin dependencias nuevas.

---

## 2026-09-02 — `Inference/evaluate.py` (+ protocolo del ciclo test)

**Estado: HECHO.** Evaluación del sistema expuesto sobre las filas
`ml_estado=='test'`, verificada en sandbox.

### Protocolo consensuado del ciclo (la parte que tocaba a este archivo)

- Estado estable tras reentrenar: train + filtered + abiertas, **ninguna
  test**. Las test solo existen entre update y reentrenamiento.
- `Scraper/update.py` (pendiente) marcará 'test' toda fila **recién
  adjudicada que no pertenezca a train** — incluye refrescos que cierran
  expedientes y filas nuevas que llegan ya adjudicadas.
- evaluate.py re-infiere el conjunto test con `Models/` en cada evaluación
  (ambos modos) vía `inferir()`; **no escribe preds en el parquet** (las
  columnas pred son el registro de servido y esas filas no fueron servidas).
- Validez: una fila test es evaluable si clasificaría como 'train' — se
  reutiliza `cleaning.clasificar_estado` sin duplicar criterios. Las
  inválidas quedan para que cleaning las reclasifique.
- **Promoción**: `--modo prepromote` registra la evaluación de cierre del
  modelo que va a ser reemplazado (la versión del registro es el
  `max meta['creado']` vigente ANTES de reentrenar). Después:
  cleaning (sin cambios — ya pliega test→train al recalcular) → featurer →
  training. El modelo nuevo cierra su evaluación cuando le toque ser
  reemplazado.
- Registro en `<data-dir>/evaluaciones/` (cada host sus datos; gitignored):
  `estado_curso.json` (modo curso, se sobreescribe — dashboard del modelo
  expuesto) + `historico.jsonl` (modo prepromote, append — histórico).

### Qué mide

Por conjunto (y sobre la población válida por línea): num MAE · zd AUC-PR +
precision/recall al umbral del router + prevalencia · disc MAE · system MAE
(gate). Registro JSON: modo, fecha, versión de modelo, umbrales, recuentos
(test/evaluadas/excluidas), ventana de publicación.

### Verificación

Sandbox con estado post-update simulado: 2.000 filas marcadas 'test'
(1.600 válidas + 400 filtered) + 100 control sin marcar.

- Solo lee test (las 100 de control fuera); las 400 inválidas excluidas por
  el filtro de cleaning; 1.600 evaluadas.
- curso sobrescribe `estado_curso.json`; prepromote añade una línea por run
  (2 runs → 2 líneas parseables en `historico.jsonl`); dry-run no registra.
- El parquet queda intacto (md5 verificado): evaluate nunca escribe el
  almacén.
- Métricas del smoke coherentes (muestras de train, in-sample orientativo):
  lic num 0,78 · zd AUC-PR 0,97 · disc 6,66 · system 6,59 | men 0,18 ·
  0,998 · 0,87 · 0,84. El gate mejora el disc en ambos conjuntos.

### Operativa

- Ejecutar: `.venv/bin/python -u Inference/evaluate.py [--modo
  curso|prepromote] [--data-dir Data] [--models-dir Models] [--feats
  Data/features.parquet] [--dry-run]`
- Coste: el de un `inferir()` del conjunto test (~14 s para 1,6k filas).
- Sin dependencias nuevas (sklearn ya estaba).

---

## 2026-09-02 — `Scraper/update.py`: marca de 'test'

**Estado: HECHO** (camino positivo pendiente de transiciones reales).

Qué hace: la regla del protocolo del ciclo de evaluación — en el merge, toda
fila **recién adjudicada** que no sea train pasa a `ml_estado='test'`.
"Recién adjudicada" = fila nueva que llega cumpliendo la condición, o id
conocido que antes no la cumplía. La condición espeja la `cerrada` de
cleaning.py: licitaciones `estado ∈ {Resuelta, Adjudicada}`; menores
`num_ofertas` o `importe_adjudicacion` presentes. Las test existentes no se
desmarcan; las train nunca se tocan; preds/version se siguen conservando.
Stats del merge con contador propio (`test`).

### Run real (2026-09-02, segundo update del día)

0 nuevos · 0 cambiados · **0 marcadas test** en ambos conjuntos: el primer
update del día (12:19) ya ingirió la novedad y PLACSP no movió nada en 4 h.
Verificación contra pre-captura del estado: 3.630.925 filas, 0 ids
duplicados, esquema == FINAL_SCHEMA, `ml_estado` idéntico en todos los ids,
0 test. **El camino transición→marca quedó sin ejercitar por datos reales**
(hoy no las hubo); se estrenará en el primer update con adjudicaciones
nuevas.

### Operativa

- Sin cambios de uso: `.venv/bin/python -u Scraper/update.py [--meses N]`.

---

## 2026-09-02 — `api.py` + `Dashboard/dashboard.html`

**Estado: HECHO.** API de operación y dashboard portados de
`../Licitaciones/spts/{api,dashboard}.py|html` y adaptados al rework:
4 paneles (Datos · Laboratorio · Histórico · Modelo expuesto), verificados
en vivo contra `Data/` y `Models/` reales.

### `api.py` (raíz)

FastAPI localhost por diseño, sin registry de boosters (el servido a
demanda es el diseño de Inference/inference.py: carga y descarga por
llamada) y sin config files (constantes en el archivo):

- `GET /` — sirve `Dashboard/dashboard.html` (un solo fichero, sin build).
- `GET /status` — universo por conjunto (train/test/filtered/abiertas/
  servidas), versión expuesta, op en marcha, cola de runs. Para el poll
  de 10 s del dashboard.
- `GET /datos` — análisis del parquet: KPIs, universo, filas por año de
  publicación (pre-2021 en un bucket `<2021`), calidad media, preds por
  versión. **Cacheado por (mtime, tamaño)**: releer los 600 MB solo
  cuando update.py reescribe el almacén.
- `GET /modelos` — metas de las 6 líneas (nunca los boosters): receta HP,
  rounds, tamaños, features+TE, ventana early-stop, MB del .ubj; versión
  expuesta = `inference.version_modelos` y umbrales del router leídos de
  `inference.py` (sin duplicar constantes).
- `GET /evaluaciones` — `estado_curso.json` (curso, se sobreescribe) +
  `historico.jsonl` (cierres prepromote) de `Data/evaluaciones/`, todo
  en una respuesta (los docs son pequeños).
- `POST /ops/evaluar[?modo]` — subprocess single-flight de
  `evaluate.py` (default curso); logs en `Data/ops/logs/`, una línea por
  run en `Data/ops/runs.jsonl`, estado en memoria del proceso.
- `POST /inferir` — EL producto: upload parquet raw (+`conjunto`) →
  passthrough completo con las 5 preds + version (`inferir()`); tope
  256 MB; headers `X-Modelo-Version`/`X-Filas`/`X-Caveat`.

### `Dashboard/dashboard.html`

Mismo sistema visual del viejo (CSS, tabs con deep-link `#hash`, tiles
KPI, tablas, SVG inline, dark mode, tooltips nativos), etiquetas en
español. Los 4 paneles:

1. **Datos** — análisis del parquet: KPIs (filas, conjuntos, test pool,
   servidas, modificado), barras apiladas por ciclo de vida + tabla
   gemela, barras por año de publicación por conjunto + tabla gemela,
   preds por versión si existen.
2. **Laboratorio** — vacío por ahora; destinado a información de
   `../Licitaciones-Lab`.
3. **Histórico** — cierres prepromote de modelos ya reemplazados:
   select de versión, KPIs por conjunto (system/num/zd/disc) y tendencia
   del system MAE entre versiones (línea con ≥2 puntos).
4. **Modelo expuesto** — KPIs (versión, edad, umbrales router, pool),
   tablas por conjunto de las 3 líneas con su receta, evaluación en
   curso (`estado_curso.json`), botón «Evaluar ahora» y runs recientes.

Ciclo de refresco: `/status` cada 10 s; `/datos` `/modelos`
`/evaluaciones` al cargar y al terminar un op; estados vacíos con
instrucciones cuando aún no hay evaluaciones.

### Verificación

- Endpoints en vivo (uvicorn 127.0.0.1): `/status` `/datos` `/modelos`
  `/evaluaciones` `/` con datos reales (3.630.925 filas; 6 metas;
  evaluaciones vacías — el ciclo aún no ha producido).
- `POST /ops/evaluar` end-to-end: subprocess rc 0 («sin filas test»,
  lo esperado hoy), `runs.jsonl` con la línea del run, estado limpio;
  single-flight 409 con doble POST; `?modo=bogus` 400.
- `POST /inferir` smoke: 80 abiertas (40+40) → +6 columnas, preds sin
  NaN, gate consistente (system=0 ⇔ zd_pred), versión única; 422 con
  basura, 409 sin `conjunto`.
- Dashboard: sintaxis JS (node) + cross-check de IDs + **smoke DOM en
  node** con stub mínimo y datos reales + evaluaciones sintéticas
  (esquema de evaluate.py): ejercita los 4 paneles, select histórico,
  tendencia con 2 puntos y los caminos vacíos. El smoke cazó y corrigió
  una carrera real (los KPIs del expuesto se pintaban antes de
  `/modelos` con los umbrales a "—": ahora `cargarModelos` re-renderiza).
- Los gráficos portan el sistema visual validado del viejo (misma paleta
  por conjunto); tooltips nativos `title` en barras y puntos.

### Operativa

- Levantar: `.venv/bin/uvicorn api:app --port 8000` (desde la raíz) →
  dashboard en `http://127.0.0.1:8000/` (deep-links `#lab #historico
  #expuesto`).
- `Data/ops/` (runs + logs) es gitignored con el resto de `Data/`.
- requirements: añadidos fastapi 0.141.1, uvicorn 0.52.4,
  python-multipart 0.0.32.

---

## 2026-09-03 — `api.py` + `Dashboard/`: la API como única puerta de operación

**Estado: HECHO.** Se barajó un CLI y se descartó: el single-flight de la
API solo cubre lo que pasa por ella — con una sola puerta, la limitación
"dos update.py a la vez" se disuelve en vez de documentarse; lo scriptable
queda cubierto con `curl -X POST localhost:8000/ops/{op}`. Todo el trabajo
es ampliar el `/ops` y el dashboard para gestionar el pipeline entero.

### `api.py`

- `POST /ops/{op}` — evaluar | update | cleaning | featurer | training |
  **ciclo**. Single-flight como antes; query `modo` (evaluar suelto:
  curso|prepromote) y `meses` (update y ciclo). scrape queda fuera
  (reconstruir el almacén entero es cosa de terminal); servir sigue siendo
  `POST /inferir`.
- **`ciclo`** — la cadena del protocolo de reentrenamiento: update →
  evaluar prepromote (cierra el modelo que va a ser reemplazado) →
  cleaning (pliega las test a train) → featurer → training. Cada paso es
  su subprocess con su log y su línea en runs.jsonl (campo `ciclo` con el
  id del grupo); **para al primer rc≠0**. Con 0 filas test, evaluar es
  no-op limpio (rc 0 sin registrar) y el ciclo sigue su curso.
- `GET /ops/log/{op}/{run_id}` — últimas N líneas del log de un run (el
  visor del dashboard). Nombre exacto `{op}_{run_id}.log`, **sin glob**:
  dos pasos de un ciclo pueden caer en el mismo segundo y compartir
  run_id (lo cazó el smoke). `op`/`run_id` validados por regex.
- `/status` extiende el estado del op con `paso`/`pasos` (progreso del
  ciclo) y `log_run_id` (el run cuyo log está vivo ahora).
- Implementación: `_start_op(op, pasos)` con la lista de pasos
  construida por `_pasos_de` (función pura, testeable); el watcher corre
  `subprocess.call` secuencial y actualiza estado/registro por paso.
- Limpieza menor: `_META_CACHE` estaba definido dos veces.

### `Dashboard/dashboard.html`

- Tarjeta **«Ciclo completo»** (input de meses + botón) y tarjeta
  **«Pasos sueltos»** (update/cleaning/featurer/training con sus
  defaults) junto a la de «Evaluar ahora»; todos los botones se
  deshabilitan con un op en marcha.
- **Tira de estado** del op en marcha con chips de pasos (✓ ok · ● en
  marcha · ✗ fallido · · pendiente) cuando es un ciclo.
- **Runs recientes**: los pasos de ciclo llevan marca ⤷ (tooltip con el
  id del grupo) y cada fila tiene botón **log**.
- **Visor de log** bajo la tabla: abre/cierra por fila y el del paso en
  marcha se auto-refresca con el poll de 10 s.

### Verificación

- Cadena con stubs (sandbox `/tmp` para no tocar `Data/ops/` real):
  `_pasos_de` construye bien (prepromote forzado en el ciclo, `--meses`
  solo en update); 2 pasos ok → fallo rc 3 → el paso posterior **no
  corre**; runs.jsonl agrupa por campo `ciclo`; un log por paso; el op
  suelto comparte run_id entre respuesta/estado/registro; el visor
  resuelve el log correcto incluso con colisión de segundo; sanitizadores
  devuelven 400/404 con entradas maliciosas.
- HTTP en vivo (uvicorn): `POST /ops/evaluar` real rc 0 («sin filas
  test», lo esperado hoy); 409 con doble POST; 404 op desconocido; 400
  modo/meses/run_id; visor del run recién creado.
- Dashboard: `node --check` del script + smoke DOM con stubs (eval
  indirecto para replicar el scope global del navegador): chips del
  ciclo, marca ⤷ y botones de log, visor abre/pinta/cierra, feedback
  ok/err del lanzamiento, cross-check de IDs y handlers.

### Primer ciclo real (mismo día, estreno del protocolo)

Un `POST /ops/update?meses=2` encontró novedades (PLACSP movió): 170+144
nuevas, **614 marcadas test** — el pool de evaluación se estrenó. Un
`POST /ops/evaluar` (curso) registró la primera `estado_curso.json`, y a
continuación el **primer ciclo completo por la API** (31,5 min):

| paso | duración | resultado |
|---|---|---|
| update | 3:17 | 0/0/0 — nada nuevo en los 15 min previos |
| evaluar prepromote | 0:21 | **primer cierre en `historico.jsonl`**: v2026-09-02 sobre las 614 test (606 eval), números idénticos al curso previo (determinismo) |
| cleaning | 0:11 | pliegue: lic +458 train (12 a filtered), men +144, test→0; totales por conjunto intactos |
| featurer | 1:27 | `features.parquet` regenerado con el train nuevo |
| training | 26:25 | 6 líneas (rounds 991/346/786/1459/392/233) |

- Nueva versión expuesta: **2026-09-03T12:16:31Z**; los 5 runs del ciclo
  quedaron agrupados por el campo `ciclo` en `runs.jsonl`.
- Smoke de servido post-ciclo (`POST /inferir`, 100 abiertas): versión
  nueva en headers, 5 preds sanas, gate consistente (`system==0 ⇔
  zd==1 | disc==0`).
- Cierre honesto del modelo 2026-09-02 (sus números sobre las 614): lic
  system MAE 10,53 · num 1,36 · zd AUC-PR 0,693 · disc 10,52 | men 3,71
  · 0,58 · 0,954 · 3,72. Peor que las referencias VAL (optimistas por
  diseño: entraban en el refit) y con n=606 — primera línea base real;
  el panel Histórico empezará la tendencia con este punto.
- **Nota de protocolo**: tras un ciclo, `estado_curso.json` conserva la
  evaluación del modelo reemplazado hasta que acumulen nuevas test y se
  lance un curso — el panel lo hace legible mostrando la versión
  evaluada dentro del propio documento.

### Operativa

- Sin cambios de arranque ni dependencias nuevas:
  `.venv/bin/uvicorn api:app --port 8000` → dashboard en `/`.
- Cron-able: `curl -s -X POST 'http://127.0.0.1:8000/ops/update?meses=3'`.

---

## 2026-09-05 — Puerto de los campeones de Nueva_Licitaciones_Lab

**Estado: HECHO.** El puerto a mano que el charter del Lab define: sus
campeones (BUILDLOG sesión 9, 86 corridas registradas) traducidos a las
constantes de `Modeling/`, el panel «Laboratorio» del dashboard poblado con
su registro (pendiente #2 de este BUILDLOG), y el run real completo que
estrenó los modelos nuevos. Versión expuesta:
**2026-09-05T11:27:41Z**.

### Campeones promovidos (specs en `experimentos/` del Lab)

| Línea | Experimento | Encoding órgano | Receta | best_it Lab |
|---|---|---|---|---|
| lic/num | `xgb_d8_eta01_mae_teorg_m100_t3` | TE m=100 (reemplaza) | absoluteerror d8/eta0.1, 18 feats | 389 |
| lic/zd | `xgb_d8_eta01_freqorg_t3` | frecuencia de train (reemplaza) | binary d8/eta0.1, 18 feats | 384 |
| lic/disc | `xgb_d8_eta01_mae_teorg_m20_clip_t3` | TE m=20 (reemplaza) | absoluteerror d8/eta0.1 + clip | 394 |
| men/num | `xgb_d8_mae_teorg_m20nat` | TE m=20 extra (`f_organo_te`) + nativo | absoluteerror d8/eta0.3, 13 feats | 256 |
| men/zd | `xgb_d6_t3` | nativo | binary d6/eta0.3, 17 feats | 158 |
| men/disc | `xgb_d10_mae_teorg_m20_t3` | TE m=20 (reemplaza) | absoluteerror d10/eta0.3, 17 feats | 195 |

Comunes: `hist`, seed 42, n_jobs 6, early stopping 30 rondas sobre VAL.
Desaparecen los HPs del puerto viejo (subsample/colsample/mcw/reg_lambda) y
todo `scale_pos_weight`. **Población sin cambios**: la unión `ok_*` del Lab
== `ml_estado=='train'` (paridad exacta verificada por el Lab), y el filtro
por objetivo válido de training reproduce `ok_*` por línea — `cleaning.py`
solo cambió comentarios.

### Orden de sesión: el prepromote necesita el código VIEJO

`evaluate.py` corre como subprocess e importa `featurer.py` del repo: editado
el código, la evaluación de cierre del modelo saliente rompería (metas viejas
↔ featurer nuevo). Por eso el run fue: **update + evaluar prepromote con el
código anterior** (cierra el modelo 2026-09-03) → swap de código → cleaning →
featurer → training → umbrales.

### `Modeling/featurer.py` — reescritura (19 `f_*`)

Puerto literal de `featuring.py::derivar` (mismas fórmulas, mismos nombres
`f_*`, mismos tipos Arrow — `FEAT_SCHEMA`): muere todo lo del Lab viejo
(harden 1e9, cap 10 del ratio, keywords, hist_volume_*, row_missing_count,
INT-*, fillna("missing"), drops por varianza). `featurizar_conjunto(df,
conjunto)` conserva la firma (la sigue usando inference.py sin copia);
`FEATURES_LINEA` nueva = la selección por línea de los campeones; `TIPOS`
(num|cat) es el contrato compartido con training. features.parquet:
**3.517.822 × 26** (4 claves + 3 objetivos + 19 f_*), licitaciones primero.

### `Modeling/training.py` — las 6 recetas

`LINEAS` con los campeones (encoding del órgano por línea: `te` m100 / `frecuencia`
/ `te` m20 / `te_extra` m20 / `nativo` / `te` m20). El TE es el del Lab —
`te(g) = (n_g·media_g + m·prior)/(n_g + m)` — con la **sonda** en su régimen
exacto: TRAIN con valores OOF (K=5, rng seed 42), VAL con el mapa full-train
+ prior. **Rounds = best_iteration + 1 exacto** (fuera el buffer +50).
Categorías fit-on-train **por orden de aparición** (no `sorted()`), código
−1 → NaN. Meta: mismas claves; `target_encoding` → **`encoding_organo`**
`{tipo, columna, m, prior, maps}`; `transform` siempre None (num directo,
sin log1p → adiós expm1 en servido).

Divergencias conscientes y documentadas: el refit de producción (que el Lab
no hace, sin artefactos) usa el **mapa TE re-fit sobre train+val** — el mismo
que va al meta, consistencia booster↔servido; la auto-inclusión queda
acotada por el suavizado y los rounds se congelaron con la sonda OOF. GPU
uniforme (el campeón men/num corrió en CPU; desplazamiento +0,2–0,4%
documentado en el Lab, aceptado). El clip num [0,50] en servido es política
del motor (el Lab no clipea num).

### `Inference/` — servido sin `Data/`

Sin hist_volume_* ni row_missing_count, **`inferir()` deja de necesitar
features.parquet** (fuera `feats_path`, `cargar_hist_base`, `aplicar_hist`,
`SEGS_HIST`, `--feats` y el header `X-Caveat`): el servido queda firme sin
`Data/` (verificado sirviendo con el parquet apartado). `matriz_linea`
resuelve el órgano según `encoding_organo.tipo` (TE → prior; frecuencia →
NaN sin fill; te_extra → sintetiza `f_organo_te`; nativo → niveles).
evaluate.py solo cambió de firma.

### api.py + Dashboard — `GET /laboratorio` y panel

`PROMOVIDOS_LAB` (constante de api.py: el registro del Lab NO marca
campeones, el puerto a mano actualiza la constante) + `LAB_DIR` →
`GET /laboratorio` lee los 6 `*.trainings.json` (cache por mtime): por línea
el campeón (receta, encoding, best_it, device, métricas val/test del
protocolo del Lab, mejor baseline) + resumen del registro. Panel
«Laboratorio» poblado (KPIs + tabla por conjunto); `/modelos` muestra el
encoding del órgano en la columna Features y la hp nueva (device/seed/jobs).

### El run real (vía API, orden prepromote-primero)

| paso | resultado |
|---|---|
| update `--meses 2` | 3.631.481 filas (+242): lic 157 nuevos/1.877 cambiados/**488 test** · men 85/11/**85 test** |
| evaluar prepromote | **cierre del modelo 2026-09-03** sobre 573 test (568 evaluables): lic system MAE **9,93** (num 1,36 · AUC-PR 0,783 · disc 10,10) · men **5,61** (0,69 · 0,913 · 5,62) — segunda línea del histórico |
| cleaning | pliegue test→0; totales por conjunto intactos |
| featurer | features.parquet 3.517.822 × 26 (19 f_*) |
| training | 6 líneas, ~4 min GPU: rounds **385/473/550/587/256/184** (lic/num y men/disc cerca de los best_it del Lab; el resto difiere por la ventana VAL as-of hoy vs as-of M del Lab) |
| umbrales | re-derivados sobre VAL: **lic 0,475→0,365** (system MAE 7,09→7,07 · P/R zero 0,87/0,90) · **men 0,585→0,555** (1,26→1,24 · 0,97/0,98), curvas planas; uvicorn reiniciado (la constante se lee al importar) |

Modelos: ~710 MB totales (antes ~4,6 GB); el mayor men/num 528 MB (órgano
nativo 16.940 niveles, 587 rounds d8). Referencia VAL con los boosters
nuevos (optimista por diseño — VAL entra en el refit): lic num MAE 1,04 · zd
AUC-PR 0,956 · disc 7,50 · system 7,07 | men 0,22 · 0,997 · 1,25 · 1,24. El
cierre real del modelo nuevo llegará con el próximo ciclo.

### Verificación

- **Paridad de oro del featurer**: join por id (3,5M filas) contra el
  parquet del Lab — **0 diffs inesperados** en las 19 columnas; los únicos
  diffs caen en las 821/1.167 filas cuyo raw refrescó PLACSP tras la siembra
  del Lab (clasificadas por `fecha_updated`). Idempotencia bit a bit
  (segunda corrida `equals`).
- **Poblaciones exactas** vs Lab: lic num 756.864 · men num 2.760.165 · men
  disc 2.711.647 idénticos; lic disc 608.466 = 608.462 + 4 filas nuevas del
  update. `zero_discount.notna() == discount_pct.notna()` y zd=1 ⇔ disc=0.
- **Training smoke al 5%** en sandbox: las 6 líneas encadenan
  sonda→rounds→refit→meta; metas coherentes con `LINEAS` (feats
  18/18/18/13/17/17, encodings, clips, prior/mapas plausibles).
- **Contrato de servido**: featurize de servido == tabla train (2k filas por
  conjunto, exacto); preds sanas y deterministas; matriz float32 ==
  float64 en preds (0.0e+00 de diferencia); órgano no visto → prior (TE) /
  NaN (frecuencia/nativo).
- **API/dashboard**: `/laboratorio` con el registro real (86 entradas, 6
  campeones), `/modelos` con la receta nueva, `/inferir` 200 filas smoke
  (versión nueva en headers, gate consistente, sin X-Caveat) y **firmeza
  sin `Data/features.parquet`**; `node --check` + smoke DOM del panel nuevo
  (KPIs, tablas, camino vacío) y del expuesto con el meta nuevo.
- Bug del puerto cazado por la verificación: `matriz_linea` sintetizaba la
  columna del órgano fuera de posición y XGBoost rechazaba el orden de
  features — corregido construyendo X en el orden del meta y sobrescribiendo
  in-place. (Y un `max(None, …)` en `_laboratorio` que la primera petición
  en vivo cazó.)

### Pendiente (orden propuesto, a consensuar)

1. Flujo operativo de preds — persistir lo servido en
   `licitaciones.parquet` (el merge de preds que el BUILDLOG de
   inference.py deja al llamador: qué filas se sirven, cuándo y con qué
   versión).
2. Umbrales del router derivados por ciclo — hoy son un one-off por sesión;
  automatizar su re-derivación tras cada training (opción evaluada y
  descartada esta sesión por alcance; el BUILDLOG de la próxima promoción
  decide).
