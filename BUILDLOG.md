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

## Pendiente (orden propuesto, a consensuar)

1. `Scraper/update.py` — marcar 'test' la adjudicación nueva (regla del
   protocolo de arriba). Luego cleaning tal cual pliega test→train al
   reentrenar.
2. `api.py` + flujo operativo — servir y persistir preds en
   `licitaciones.parquet` (el merge de preds es del llamador).
3. `Dashboard/` (lee `Data/evaluaciones/`).

`requirements.txt` + venv propio + git (main, `Data/` ignorado) quedaron
listos el 2026-09-02, antes de iniciar update.py.
