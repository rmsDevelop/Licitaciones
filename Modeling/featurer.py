#Crea Data/features.parquet con las filas ml_estado=='train' (ambos conjuntos):
#id + ano + fecha_publicacion (claves de split) + los tres objetivos
#(num_ofertas, discount_pct, zero_discount) + la UNION de features de las seis
#lineas del Lab. Los criterios son un puerto de Licitaciones-Lab
#(_common/featurizer.py): mismas transformaciones, mismas reglas de leakage
#(solo conocidas en publicacion), mismos casts categoricos. Donde las lineas
#del Lab discrepaban (base monetaria, caps de ratio/duracion, texto de objeto)
#la tabla union lleva variantes con sufijo y FEATURES_LINEA decide cual usa cada
#linea — ese dict es el contrato compartido con training.py e Inference/.

from __future__ import annotations

import argparse
import os
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constantes portadas del Lab
# ---------------------------------------------------------------------------
MONEY_ERR_CEILING = 1e9          # euros; por encima, error de datos -> NaN
DURATION_DAYS_MAX = 365.25 * 50  # duraciones mayores/negativas -> NaN
DURACION_CAP_DAYS = 365.25 * 10  # tope del log capado (una decada)
UNIT_TO_DAYS = {"ANN": 365.25, "MON": 30.437, "DAY": 1.0}
CPV_DELIM = ";"

# Flags de keywords del objeto (insensibles a acentos; mismo set del Lab).
OBJETO_KEYWORDS = [
    "acuerdo marco", "emergencia", "obra", "redes", "seguro", "limpieza",
    "software", "salud", "sanitario", "electric", "ingenieria", "tecnolog", "urgente",
]

# Columnas dropeadas para TODAS las lineas (puerto de COMMON_DROP_COLS):
# fugas post-adjudicacion, no conocidas en publicacion, calidad leaky,
# cuasi-constantes, identificadores/texto/precedencia y columnas raw
# reemplazadas por derivadas. INT-CONS-20 solo existe en licitaciones.
DROP_COLS = [
    # post-adjudicacion (fuga)
    "importe_adj_con_iva", "importe_adjudicacion",
    "adjudicatario", "nif_adjudicatario", "fecha_adjudicacion", "fecha_updated",
    # no conocidas en publicacion
    "estado", "estado_code",
    # calidad leaky + compuesto
    "score_calidad", "INT-VAL-02", "INT-VAL-03", "INT-VAL-07", "INT-VAL-12",
    "INT-CONS-01", "INT-CONS-08", "INT-CONS-18", "INT-CONS-20", "INT-FIA-01", "INT-FIA-09",
    # cuasi-constantes serve-safe
    "INT-VAL-06", "INT-VAL-09", "INT-VAL-10", "INT-FIA-08",
    # identificadores / texto / precedencia / reemplazadas
    "expediente", "objeto", "nif_organo", "id_plataforma", "dependencia",
    "url", "hora_limite", "es_menor",
    "valor_estimado_contrato", "importe_con_iva",
    "duracion", "duracion_unidad", "fecha_limite",
    "nuts", "cpv_principal", "cpvs",
    "procedimiento", "tipo_contrato",  # las _code las portan
    "ubicacion",  # redundante con nuts
]
# importe_sin_iva: dropeada en licitaciones; feature en menores zd/disc.

CATEGORICAL_BASE = [
    "tipo_contrato_code", "subtipo_code", "urgencia",
    "financiacion_ue", "es_pyme",
    "organo_contratante", "dir3_organo", "ciudad_organo",
    "cpv_division", "cpv_group", "cpv_class",
    "nuts_country", "nuts1", "nuts2", "nuts3",
]

# ---------------------------------------------------------------------------
# Bloques de features (composicion de las listas por linea)
# ---------------------------------------------------------------------------
FE_CPV = ["cpv_division", "cpv_group", "cpv_class", "cpv_count", "is_multi_cpv",
          "cpv_count_missing", "n_cpv_divisions", "is_multi_division"]
FE_NUTS = ["nuts_country", "nuts1", "nuts2", "nuts3", "nuts_granularity"]
FE_DURACION = ["duracion_days", "duracion_missing"]
FE_TIEMPO = ["pub_month", "pub_quarter", "pub_dayofweek", "pub_is_weekend",
             "pub_is_august", "deadline_missing"]
FE_OBJETO_BASE = ["objeto_has_" + kw.replace(" ", "_") for kw in OBJETO_KEYWORDS]
FE_OBJETO_TEXT = ["objeto_len", "objeto_word_count"]
FE_INTS = ["INT-VAL-01", "INT-VAL-04", "INT-VAL-05", "INT-VAL-14", "INT-FIA-11"]
FE_MISSING = ["valor_estimado_contrato_missing", "row_missing_count"]

# Dinero por linea (base + cap del ratio):
#   lic num/zd:  log_budget_con_iva + budget_to_estimado_con_iva_raw
#   lic disc:    log_budget_con_iva + budget_to_estimado_con_iva_cap
#   men num:     log_budget_con_iva + budget_to_estimado_con_iva_cap
#   men zd/disc: log_budget_sin_iva + budget_to_estimado_sin_iva_cap + importe_sin_iva
DINERO_LIC_RAW = ["log_budget_con_iva", "budget_to_estimado_con_iva_raw"]
DINERO_LIC_CAP = ["log_budget_con_iva", "budget_to_estimado_con_iva_cap"]
DINERO_MEN_NUM = ["log_budget_con_iva", "budget_to_estimado_con_iva_cap"]
DINERO_MEN_SIN = ["log_budget_sin_iva", "budget_to_estimado_sin_iva_cap", "importe_sin_iva"]

# Duracion log: lic num/zd sin cap (raw); el resto capado a una decada.
LOG_DUR_RAW = ["log_duracion_days_raw"]
LOG_DUR_CAP = ["log_duracion_days"]

HIST_LIC = ["hist_volume_nuts3", "hist_volume_cpv_division", "hist_volume_procedimiento_code"]
HIST_MEN = ["hist_volume_nuts3", "hist_volume_cpv_division", "hist_volume_tipo_contrato_code"]

_BASE = list(dict.fromkeys(
    ["ano"] + FE_CPV + FE_NUTS + FE_DURACION + FE_TIEMPO + FE_OBJETO_BASE + FE_INTS
    + FE_MISSING + CATEGORICAL_BASE + ["INT-FIA-04"]))

# Las seis lineas del Lab (campeones): la lista exacta de columnas que usa cada
# una. Contrato con training.py (y con Inference/, via el meta de cada modelo).
FEATURES_LINEA = {k: list(dict.fromkeys(v)) for k, v in {
    "licitaciones/num_ofertas": ["procedimiento_code"] + _BASE + DINERO_LIC_RAW + LOG_DUR_RAW + HIST_LIC,
    "licitaciones/zero_discount": ["procedimiento_code"] + _BASE + DINERO_LIC_RAW + LOG_DUR_RAW + HIST_LIC,
    "licitaciones/discount": ["procedimiento_code"] + _BASE + DINERO_LIC_CAP + LOG_DUR_CAP + HIST_LIC + FE_OBJETO_TEXT,
    "menores/num_ofertas": _BASE + DINERO_MEN_NUM + LOG_DUR_CAP + HIST_MEN + FE_OBJETO_TEXT,
    "menores/zero_discount": _BASE + DINERO_MEN_SIN + LOG_DUR_CAP + HIST_MEN + FE_OBJETO_TEXT,
    "menores/discount": _BASE + DINERO_MEN_SIN + LOG_DUR_CAP + HIST_MEN + FE_OBJETO_TEXT,
}.items()}

# Objetivo por linea y poblacion (mascara sobre la tabla de features).
TARGET_LINEA = {
    "num_ofertas": ("num_ofertas", "num_ofertas.notna() & between(0,50)"),
    "zero_discount": ("zero_discount", "discount_pct.notna() & between(0,70)"),
    "discount": ("discount_pct", "discount_pct.notna() & between(0,70)"),
}

# ---------------------------------------------------------------------------
# Helpers (puerto del Lab)
# ---------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def harden(df: pd.DataFrame) -> pd.DataFrame:
    """Tope de dinero implausible (1e9); las fechas quedan como estan."""
    df = df.copy()
    for col in ["importe_sin_iva", "importe_con_iva", "valor_estimado_contrato"]:
        if col in df.columns:
            df.loc[df[col] > MONEY_ERR_CEILING, col] = np.nan
    return df


def add_cpv_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cpv = df["cpv_principal"].astype("string")
    df["cpv_division"] = cpv.str.slice(0, 2)
    df["cpv_group"] = cpv.str.slice(0, 3)
    df["cpv_class"] = cpv.str.slice(0, 4)
    n_cpv = df["cpvs"].astype("string").str.split(CPV_DELIM).str.len()
    df["cpv_count"] = pd.to_numeric(n_cpv, errors="coerce")
    df["is_multi_cpv"] = (n_cpv.fillna(1) > 1).astype("int8")
    df["cpv_count_missing"] = n_cpv.isna().astype("int8")
    parts = df["cpvs"].astype("string").str.split(CPV_DELIM)
    n_div = parts.apply(lambda lst: len({p.strip()[:2] for p in lst
                                         if isinstance(lst, list) and len(p.strip()) >= 2})
                        if isinstance(lst, list) else np.nan)
    n_div = pd.to_numeric(n_div, errors="coerce")
    df["n_cpv_divisions"] = n_div.clip(upper=5)
    df["is_multi_division"] = (n_div.fillna(1) > 1).astype("int8")
    return df


def add_nuts_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = df["nuts"].astype("string")
    df["nuts_country"] = n.str.slice(0, 2)
    df["nuts1"] = n.where(n.str.len() >= 3).str.slice(0, 3)
    df["nuts2"] = n.where(n.str.len() >= 4).str.slice(0, 4)
    df["nuts3"] = n.where(n.str.len() >= 5).str.slice(0, 5)
    df["nuts_granularity"] = pd.to_numeric(n.str.len(), errors="coerce")
    return df


def add_money_features(df: pd.DataFrame) -> pd.DataFrame:
    """Las variantes de las seis lineas, cada una con su nombre."""
    df = df.copy()
    con, sin, valor = df["importe_con_iva"], df["importe_sin_iva"], df["valor_estimado_contrato"]
    df["log_budget_con_iva"] = np.log1p(con)
    df["log_budget_sin_iva"] = np.log1p(sin)
    ratio_con = (con / valor).replace([np.inf, -np.inf], np.nan)
    ratio_sin = (sin / valor).replace([np.inf, -np.inf], np.nan)
    df["budget_to_estimado_con_iva_raw"] = ratio_con
    df["budget_to_estimado_con_iva_cap"] = ratio_con.clip(upper=10.0)
    df["budget_to_estimado_sin_iva_cap"] = ratio_sin.clip(upper=10.0)
    return df


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    days = pd.to_numeric(df["duracion"], errors="coerce") * df["duracion_unidad"].astype("string").map(UNIT_TO_DAYS)
    days = days.where(days.between(0, DURATION_DAYS_MAX))
    df["duracion_days"] = days.astype("float64")
    df["log_duracion_days"] = np.log1p(days.clip(upper=DURACION_CAP_DAYS))
    df["log_duracion_days_raw"] = np.log1p(days)
    df["duracion_missing"] = days.isna().astype("int8")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pub = df["fecha_publicacion"]
    dow = pub.dt.dayofweek
    df["pub_month"] = pd.to_numeric(pub.dt.month, errors="coerce")
    df["pub_quarter"] = pd.to_numeric(pub.dt.quarter, errors="coerce")
    df["pub_dayofweek"] = pd.to_numeric(dow, errors="coerce")
    df["pub_is_weekend"] = (dow >= 5).astype("int8")
    df["pub_is_august"] = (pub.dt.month == 8).astype("int8")
    df["deadline_missing"] = df["fecha_limite"].isna().astype("int8")
    return df


def add_objeto_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj = df["objeto"].fillna("").str.lower().map(_strip_accents)
    df["objeto_len"] = obj.str.len().astype("Int32")
    df["objeto_word_count"] = obj.str.split().str.len().astype("Int32")
    for kw in OBJETO_KEYWORDS:
        df["objeto_has_" + kw.replace(" ", "_")] = obj.str.contains(kw, regex=False, na=False).astype("int8")
    return df


def encode_flags(df: pd.DataFrame) -> pd.DataFrame:
    """es_pyme / financiacion_ue -> categoria 'missing' explicita; INT-FIA-04 -> int."""
    df = df.copy()
    df["es_pyme"] = df["es_pyme"].map({True: "si", False: "no"}).fillna("missing")
    df["financiacion_ue"] = df["financiacion_ue"].astype("string").fillna("missing")
    if "INT-FIA-04" in df.columns:
        df["INT-FIA-04"] = df["INT-FIA-04"].astype("int8")
    return df


def add_missingness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "valor_estimado_contrato" in df.columns:
        df["valor_estimado_contrato_missing"] = df["valor_estimado_contrato"].isna().astype("int8")
    return df


def add_prior_year_volumes(df: pd.DataFrame, seg_lists: list[list[str]]) -> pd.DataFrame:
    """Recuentos del ano ANTERIOR por segmento (ano < ano de la fila): contexto
    de mercado sin fuga (nunca medias del objetivo)."""
    df = df.copy()
    for seg in seg_lists:
        name = "hist_volume_" + seg[0]
        g = df.groupby(seg + ["ano"], observed=True).size()
        prior = g.groupby(level=seg).cumsum() - g
        idx = pd.MultiIndex.from_frame(df[seg + ["ano"]])
        df[name] = pd.to_numeric(prior.reindex(idx).to_numpy(), errors="coerce")
    return df


def fix_ano(s: pd.Series) -> pd.Series:
    """Repara el bug de anno a dos digitos y acota a (2000, ano en curso)."""
    s = s.copy()
    two_digit = s.between(0, 99)
    s = s.where(~two_digit, s + 2000)
    s = s.where(s.between(2000, pd.Timestamp.now().year))
    return s


# ---------------------------------------------------------------------------
# Featurizacion de un conjunto
# ---------------------------------------------------------------------------
def featurizar_conjunto(df: pd.DataFrame, conjunto: str) -> pd.DataFrame:
    """Tabla de features union para las filas train de un conjunto.

    Recibe el DataFrame con las columnas raw + calidad (sin preds/ml_estado)
    ya filtrado a ml_estado=='train'. Devuelve id/claves/objetivos + features.
    """
    es_lic = conjunto == "licitaciones"

    # 0. Objetivos. discount_pct por base del conjunto; zero_discount = (disc==0).
    if es_lic:
        base, adj = df["importe_con_iva"], df["importe_adj_con_iva"]
    else:
        base, adj = df["importe_sin_iva"], df["importe_adjudicacion"]
    disc = (1 - adj / base) * 100
    df = df.assign(
        discount_pct=disc.where(base.notna() & (base > 0) & adj.notna()),
    )
    df["zero_discount"] = (df["discount_pct"] == 0).astype("float32").where(df["discount_pct"].notna())

    # 1. Missingness de fila: sobre las raw + calidad + objetivos (las preds y
    #    ml_estado no describen la fila y no cuentan).
    conteo_cols = [c for c in df.columns if c not in ("ml_estado",)]
    df["row_missing_count"] = df[conteo_cols].isna().sum(axis=1).astype("float64")

    # 2. Derivadas (puerto del Lab).
    df = harden(df)
    df = add_cpv_features(df)
    df = add_nuts_features(df)
    df = add_money_features(df)
    df = add_duration_features(df)
    df = add_time_features(df)
    df = add_objeto_flags(df)
    df = encode_flags(df)
    df = add_missingness(df)
    df = add_prior_year_volumes(df, [["nuts3"], ["cpv_division"],
                                     ["procedimiento_code"] if es_lic else ["tipo_contrato_code"]])

    # 3. Drops: lista comun + importe_sin_iva en licitaciones (en menores es
    #    feature de zd/disc); procedimiento_code en menores (constante).
    drops = list(DROP_COLS)
    if es_lic:
        drops.append("importe_sin_iva")
    else:
        drops.append("procedimiento_code")
    df = df.drop(columns=[c for c in drops if c in df.columns])

    # 4. Cast categorico (missing explicito).
    for c in CATEGORICAL_BASE + (["procedimiento_code"] if es_lic else []):
        df[c] = df[c].astype("string").fillna("missing").astype("category")

    return df


CLAVES = ["id", "conjunto", "ano", "fecha_publicacion"]
OBJETIVOS = ["num_ofertas", "discount_pct", "zero_discount"]


def run(data_dir: Path, salida: Path | None = None) -> None:
    path = data_dir / "licitaciones.parquet"
    salida = salida or data_dir / "features.parquet"

    # Solo las columnas necesarias (raw + calidad + estado de ML para filtrar).
    cols = pq.read_schema(path).names
    leer = [c for c in cols if c not in
            ("num_ofertas_pred", "zero_discount_prob", "zero_discount_pred",
             "discount_pct_pred", "system_discount_pct_pred", "version")]
    df = pq.read_table(path, columns=leer).to_pandas()
    df = df[df["ml_estado"] == "train"].drop(columns=["ml_estado"]).reset_index(drop=True)
    df["ano"] = fix_ano(df["ano"]).round().astype("Int64")
    print(f"filas train: {len(df):,}")

    tablas = []
    for conjunto in ("licitaciones", "menores"):
        sub = df[df["conjunto"] == conjunto].copy()
        print(f"featurizando {conjunto}: {len(sub):,} filas")
        fe = featurizar_conjunto(sub, conjunto)

        # Features union del conjunto = columnas presentes menos claves/objetivos.
        feats = [c for c in fe.columns if c not in CLAVES + OBJETIVOS]
        # Cuasi-constantes fuera (decision de tabla de entrenamiento).
        const = [c for c in feats if fe[c].nunique(dropna=True) <= 1]
        if const:
            print(f"  columnas constantes dropeadas: {const}")
            fe = fe.drop(columns=const)
            feats = [c for c in feats if c not in const]

        # Validacion del contrato: cada linea tiene todas sus features (se
        # toleran las dropeadas por constantes — sin varianza, sin senal; el
        # Lab las dropeaba igual, por linea).
        for linea in FEATURES_LINEA:
            if not linea.startswith(conjunto):
                continue
            faltan = [c for c in FEATURES_LINEA[linea]
                      if c not in fe.columns and c not in const]
            if faltan:
                raise ValueError(f"{linea}: faltan features {faltan}")

        tablas.append(fe[CLAVES + feats + OBJETIVOS])
        print(f"  {len(feats)} features, {len(fe):,} filas")
        del sub, fe

    # Union en un solo parquet (licitaciones primero, para que el filtro por
    # conjunto en training pueda saltarse row groups).
    union = pd.concat(tablas, ignore_index=True)
    del tablas, df
    print(f"escrito: {salida}  ({len(union):,} filas x {union.shape[1]} columnas)")
    tmp = salida.with_suffix(".parquet.tmp")
    union.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, salida)


def main() -> None:
    p = argparse.ArgumentParser(description="Generar features.parquet (filas train).")
    p.add_argument("--data-dir", default="Data", help="directorio con licitaciones.parquet")
    p.add_argument("--salida", default=None, help="ruta de salida del parquet")
    args = p.parse_args()
    run(Path(args.data_dir), Path(args.salida) if args.salida else None)


if __name__ == "__main__":
    main()
