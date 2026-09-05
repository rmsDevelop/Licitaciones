#Crea Data/features.parquet con las filas ml_estado=='train' (ambos conjuntos):
#claves + los tres objetivos + las 19 features f_* de Licitaciones-Lab
#(featuring.py, puerto literal de sus formulas — la verificacion de oro es un
#join por id contra el parquet del Lab exigiendo igualdad exacta). La capa es
#el conjunto CERRADO que training consume: cada linea selecciona las suyas de
#FEATURES_LINEA y el encoding del organo (TE / frecuencia / nativo) lo anade
#training — lo aprendido del split nunca es columna estatica.
#
#Divergencias conscientes respecto del featurizer anterior (Lab viejo),
#heredadas del Lab nuevo y documentadas en su BUILDLOG: SIN tepe 1e9 del
#dinero (log1p maneja la escala), SIN cap 10 del ratio al estimado (misma
#base sin-IVA en ambos operandos), f_objeto_len del texto CRUDO (sin
#NFKD-strip), nulos CONSERVADOS (adios al fillna("missing") — el fill es
#politica del trainer), f_procedimiento materializada en ambos conjuntos
#(constante '6' en menores; la seleccion es de cada linea), sin hist_volume_*
#ni row_missing_count ni keywords ni INT-*.

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# La capa f_* (puerto de FEAT_COLS/FEAT_SCHEMA de featuring.py del Lab)
# ---------------------------------------------------------------------------
FEAT_COLS = ["f_log_importe_con_iva", "f_log_importe_sin_iva", "f_ano",
             "f_pub_mes", "f_nuts1", "f_nuts3", "f_duracion_days",
             "f_duracion_missing", "f_cpv_division", "f_tipo_contrato",
             "f_procedimiento", "f_organo_contratante", "f_urgencia",
             "f_es_pyme",
             # tanda 3
             "f_cpv_grupo", "f_nuts2", "f_objeto_len",
             "f_ratio_estimado", "f_estimado_missing"]

# num | cat: el contrato de tipos que lee training (las categoricas entran
# nativas a XGBoost; los trainers y el meta deciden niveles).
TIPOS = {
    "f_log_importe_con_iva": "num", "f_log_importe_sin_iva": "num",
    "f_ano": "num", "f_pub_mes": "num", "f_duracion_days": "num",
    "f_duracion_missing": "num", "f_es_pyme": "num", "f_objeto_len": "num",
    "f_ratio_estimado": "num", "f_estimado_missing": "num",
    "f_nuts1": "cat", "f_nuts3": "cat", "f_nuts2": "cat",
    "f_cpv_division": "cat", "f_cpv_grupo": "cat", "f_tipo_contrato": "cat",
    "f_procedimiento": "cat", "f_organo_contratante": "cat", "f_urgencia": "cat",
}

# Tipos Arrow de la capa (los mismos del Lab: el parquet del motor y el suyo
# quedan comparables columna a columna).
FEAT_SCHEMA = pa.schema([
    ("f_log_importe_con_iva", pa.float64()),
    ("f_log_importe_sin_iva", pa.float64()),
    ("f_ano", pa.int16()),
    ("f_pub_mes", pa.int8()),
    ("f_nuts1", pa.string()),
    ("f_nuts3", pa.string()),
    ("f_duracion_days", pa.float64()),
    ("f_duracion_missing", pa.int8()),
    ("f_cpv_division", pa.string()),
    ("f_tipo_contrato", pa.string()),
    ("f_procedimiento", pa.string()),
    ("f_organo_contratante", pa.string()),
    ("f_urgencia", pa.string()),
    ("f_es_pyme", pa.bool_()),
    ("f_cpv_grupo", pa.string()),
    ("f_nuts2", pa.string()),
    ("f_objeto_len", pa.int32()),
    ("f_ratio_estimado", pa.float64()),
    ("f_estimado_missing", pa.int8()),
])

# Columnas raw que hacen falta para derivar (lectura ligera del almacén).
COLS_NECESARIAS = ["importe_con_iva", "importe_sin_iva",
                   "valor_estimado_contrato", "objeto", "fecha_publicacion",
                   "nuts", "duracion", "duracion_unidad", "cpv_principal",
                   "tipo_contrato_code", "procedimiento_code",
                   "organo_contratante", "urgencia", "es_pyme", "ano"]

# Duracion normalizada a dias: factores y rango validos (paridad Lab).
DURACION_FACTORES = {"ANN": 365.25, "MON": 30.437, "DAY": 1.0}
DURACION_DAYS_MAX = 365.25 * 50

# ---------------------------------------------------------------------------
# Seleccion por linea (los campeones del Lab, BUILDLOG sesion 9)
# ---------------------------------------------------------------------------
# lic: 13 base (con f_procedimiento y base CON IVA) + 5 de la tanda 3;
#      el organo lo codifica training (TE en num/disc, frecuencia en zd).
# men: 12 base (sin f_procedimiento, base SIN IVA); zd/disc anaden la tanda 3
#      y num conserva el campeón de la sesión 7 (sin tanda 3) + f_organo_te.
_LIC_T3 = ["f_cpv_grupo", "f_nuts2", "f_objeto_len", "f_ratio_estimado",
           "f_estimado_missing"]
_MEN_BASE = ["f_log_importe_sin_iva", "f_ano", "f_pub_mes",
             "f_duracion_days", "f_duracion_missing", "f_es_pyme",
             "f_nuts1", "f_nuts3", "f_cpv_division", "f_tipo_contrato",
             "f_organo_contratante", "f_urgencia"]

FEATURES_LINEA = {
    "licitaciones/num_ofertas": ["f_log_importe_con_iva", "f_ano", "f_pub_mes",
        "f_duracion_days", "f_duracion_missing", "f_es_pyme", "f_nuts1",
        "f_nuts3", "f_cpv_division", "f_tipo_contrato", "f_procedimiento",
        "f_organo_contratante", "f_urgencia"] + _LIC_T3,
    "licitaciones/zero_discount": ["f_log_importe_con_iva", "f_ano", "f_pub_mes",
        "f_duracion_days", "f_duracion_missing", "f_es_pyme", "f_nuts1",
        "f_nuts3", "f_cpv_division", "f_tipo_contrato", "f_procedimiento",
        "f_organo_contratante", "f_urgencia"] + _LIC_T3,
    "licitaciones/discount": ["f_log_importe_con_iva", "f_ano", "f_pub_mes",
        "f_duracion_days", "f_duracion_missing", "f_es_pyme", "f_nuts1",
        "f_nuts3", "f_cpv_division", "f_tipo_contrato", "f_procedimiento",
        "f_organo_contratante", "f_urgencia"] + _LIC_T3,
    "menores/num_ofertas": list(_MEN_BASE),
    "menores/zero_discount": _MEN_BASE + _LIC_T3,
    "menores/discount": _MEN_BASE + _LIC_T3,
}

# Objetivo por linea y poblacion (mascara sobre la tabla de features). La
# poblacion por linea reproduce las ok_* del Lab (paridad verificada).
TARGET_LINEA = {
    "num_ofertas": ("num_ofertas", "num_ofertas.notna() & between(0,50)"),
    "zero_discount": ("zero_discount", "discount_pct.notna() & between(0,70)"),
    "discount": ("discount_pct", "discount_pct.notna() & between(0,70)"),
}

CLAVES = ["id", "conjunto", "ano", "fecha_publicacion"]
OBJETIVOS = ["num_ofertas", "discount_pct", "zero_discount"]

# Esquema completo del parquet de entrenamiento (claves + capa + objetivos).
SALIDA_SCHEMA = pa.schema(
    [("id", pa.string()), ("conjunto", pa.string()), ("ano", pa.int64()),
     ("fecha_publicacion", pa.timestamp("us"))]
    + list(FEAT_SCHEMA)
    + [("num_ofertas", pa.float64()), ("discount_pct", pa.float64()),
       ("zero_discount", pa.float32())])


# ---------------------------------------------------------------------------
# Derivacion (puerto literal de featuring.py::derivar — formulas exactas)
# ---------------------------------------------------------------------------
def fix_ano(s: pd.Series) -> pd.Series:
    """Repara el bug de anno a dos digitos y acota a (2000, ano en curso)."""
    s = s.copy()
    two_digit = s.between(0, 99)
    s = s.where(~two_digit, s + 2000)
    s = s.where(s.between(2000, pd.Timestamp.now().year))
    return s


def derivar(d: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las columnas de FEAT_COLS para `d` (trae COLS_NECESARIAS).

    Funciones puras y vectorizadas: solo informacion conocida en publicacion,
    nulos conservados (el fill es politica del trainer).
    """
    out = pd.DataFrame(index=d.index)

    # dinero: las dos bases en log1p, sin tepe (decision del Lab)
    out["f_log_importe_con_iva"] = np.log1p(d["importe_con_iva"])
    out["f_log_importe_sin_iva"] = np.log1p(d["importe_sin_iva"])

    # tiempo: mes de publicacion (NaT -> null)
    out["f_pub_mes"] = d["fecha_publicacion"].dt.month.astype("Int8")

    # geo: slices de nuts solo si el codigo llega a ese nivel
    nuts = d["nuts"].astype("string")
    out["f_nuts1"] = nuts.where(nuts.str.len() >= 3).str[:3]
    out["f_nuts3"] = nuts.where(nuts.str.len() >= 5).str[:5]

    # duracion normalizada a dias; fuera de rango -> NaN (y missing=1)
    num = pd.to_numeric(d["duracion"], errors="coerce")
    factor = d["duracion_unidad"].astype("string").map(DURACION_FACTORES)
    days = num * factor
    out["f_duracion_days"] = days.where(days.between(0, DURACION_DAYS_MAX)).astype("float64")
    out["f_duracion_missing"] = out["f_duracion_days"].isna().astype("int8")

    # cpv: division (2 digitos) y grupo (3); nulos conservados
    out["f_cpv_division"] = d["cpv_principal"].astype("string").str[:2]
    out["f_cpv_grupo"] = d["cpv_principal"].astype("string").str[:3]

    # tanda 3: geo intermedia, texto crudo, planificacion
    out["f_nuts2"] = nuts.where(nuts.str.len() >= 4).str[:4]
    out["f_objeto_len"] = d["objeto"].str.len().astype("Int32")
    est = pd.to_numeric(d["valor_estimado_contrato"], errors="coerce")
    out["f_ratio_estimado"] = (d["importe_sin_iva"] / est).replace(
        [np.inf, -np.inf], np.nan).astype("float64")
    out["f_estimado_missing"] = est.isna().astype("int8")

    # materializaciones: copias con cast ligero, nulos conservados
    out["f_tipo_contrato"] = d["tipo_contrato_code"].astype("string")
    out["f_procedimiento"] = d["procedimiento_code"].astype("string")
    out["f_urgencia"] = d["urgencia"].astype("string")
    out["f_es_pyme"] = d["es_pyme"].astype("boolean")
    out["f_organo_contratante"] = d["organo_contratante"].astype("string")

    # tendencia: ano de publicacion (== raw ano; verificado en el Lab)
    out["f_ano"] = d["fecha_publicacion"].dt.year.astype("Int16")

    return out[FEAT_COLS]


# ---------------------------------------------------------------------------
# Featurizacion de un conjunto
# ---------------------------------------------------------------------------
def featurizar_conjunto(df: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Tabla claves + objetivos + capa f_* para las filas de un conjunto.

    Recibe el DataFrame con las columnas raw + calidad (sin preds/ml_estado)
    de UN conjunto. La usa run() (entrenamiento) e Inference/inference.py
    (servido) — mismo codigo, sin copia.
    """
    df = df.copy()
    df["ano"] = fix_ano(df["ano"]).round().astype("Int64")

    # 0. Objetivos. discount_pct por base del conjunto; zero_discount = (disc==0).
    if conjunto == "licitaciones":
        base, adj = df["importe_con_iva"], df["importe_adj_con_iva"]
    else:
        base, adj = df["importe_sin_iva"], df["importe_adjudicacion"]
    disc = (1 - adj / base) * 100
    df["discount_pct"] = disc.where(base.notna() & (base > 0) & adj.notna())
    df["zero_discount"] = (df["discount_pct"] == 0).astype("float32").where(
        df["discount_pct"].notna())

    # 1. Capa f_* (la unica feature que existe: conjunto cerrado).
    f = derivar(df)

    out = df[CLAVES].copy()
    for c in FEAT_COLS:
        out[c] = f[c]
    for c in OBJETIVOS:
        if c == "num_ofertas":
            out[c] = df["num_ofertas"].astype("float64")
        else:
            out[c] = df[c]
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run(data_dir: Path, salida: Path | None = None) -> None:
    path = data_dir / "licitaciones.parquet"
    salida = salida or data_dir / "features.parquet"

    # Lectura ligera: claves + estado + raws que alimentan la capa y objetivos.
    leer = list(dict.fromkeys(
        CLAVES + ["ml_estado"] + COLS_NECESARIAS
        + ["num_ofertas", "importe_adj_con_iva", "importe_adjudicacion"]))
    df = pq.read_table(path, columns=leer).to_pandas()
    df = df[df["ml_estado"] == "train"].drop(columns=["ml_estado"]).reset_index(drop=True)
    print(f"filas train: {len(df):,}")

    tablas = []
    for conjunto in ("licitaciones", "menores"):
        sub = df[df["conjunto"] == conjunto]
        print(f"featurizando {conjunto}: {len(sub):,} filas")
        fe = featurizar_conjunto(sub, conjunto)

        # Validacion del contrato: cada linea tiene todas sus features.
        for linea in FEATURES_LINEA:
            if linea.startswith(conjunto):
                faltan = [c for c in FEATURES_LINEA[linea] if c not in fe.columns]
                if faltan:
                    raise ValueError(f"{linea}: faltan features {faltan}")
        tablas.append(fe)
        print(f"  {len(FEAT_COLS)} features, {len(fe):,} filas")
        del sub, fe

    # Union en un solo parquet (licitaciones primero, para que el filtro por
    # conjunto en training pueda saltarse row groups) con esquema explicito.
    union = pd.concat(tablas, ignore_index=True)
    del tablas, df
    tabla = pa.Table.from_pandas(union, schema=SALIDA_SCHEMA, preserve_index=False)
    print(f"escrito: {salida}  ({len(union):,} filas x {tabla.num_columns} columnas)")
    tmp = salida.with_suffix(".parquet.tmp")
    pq.write_table(tabla, tmp, compression="snappy")
    os.replace(tmp, salida)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generar features.parquet (filas train, capa f_* del Lab).")
    p.add_argument("--data-dir", default="Data", help="directorio con licitaciones.parquet")
    p.add_argument("--salida", default=None, help="ruta de salida del parquet")
    args = p.parse_args()
    run(Path(args.data_dir), Path(args.salida) if args.salida else None)


if __name__ == "__main__":
    main()
