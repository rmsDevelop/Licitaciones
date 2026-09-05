#Clasifica las filas de licitaciones.parquet rellenando ml_estado: 'train' |
#'filtered' | null (abierta). Criterios compartidos con Nueva_Licitaciones_Lab
#(cleaning.py; la union de sus ok_* == ml_estado=='train' con paridad exacta
#verificada el 2026-09-04): ventana 2021+, fix_ano, y por conjunto las
#condiciones de objetivo. Una fila es 'train' si sirve para ALGUNO de los
#tres modelos de su conjunto (num_ofertas 0..50 registrado, o discount/zero_
#discount derivable en [0, 70] con importes sanos). Se ejecuta tras la
#evaluacion: toda fila valida pasa a 'train' (sin reserva de test); el split
#train/validation lo hace training.py. Recalcula ml_estado desde las columnas
#raw en cada ejecucion (idempotente, sin dependencia del historial).

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Criterios (compartidos con Nueva_Licitaciones_Lab)
# ---------------------------------------------------------------------------
# Rango de sanidad del raw `ano` tras reparar el bug de anno a dos digitos
# (22 -> 2022). El techo es el ano en curso: la version del Lab lo congelaba
# en 2026 porque su dataset acababa ahi.
ANO_SANITY_RANGE = (2000, date.today().year)

# Ventana de modelizacion: regimen post-COVID estable (el Lab descarta 2020
# por mezcla anormal de procedimientos).
WINDOW_YEARS = (2021, date.today().year)

# Solo los expedientes resueltos/adjudicados tienen objetivo conocido
# (licitaciones; en menores el feed ya es ~100% Resuelta y no se filtra).
AWARD_STATES = ["Resuelta", "Adjudicada"]

# num_ofertas es un recuento: [0, 50] (por encima, errores de captacion).
NUM_OFERTAS_RANGE = (0, 50)

# discount_pct = (1 - adj / presupuesto) * 100: se conservan los ceros
# (estructurales), se tiran los negativos (sobre presupuesto, implausibles)
# y se trunca a 70 (territorio de error).
DISCOUNT_RANGE = (0.0, 70.0)

# Columnas necesarias para clasificar (lectura ligera del parquet).
COLS_NECESARIAS = [
    "conjunto", "estado", "ano", "num_ofertas",
    "importe_con_iva", "importe_adj_con_iva",
    "importe_sin_iva", "importe_adjudicacion",
]


def fix_ano(s):
    """Repara el bug de anno a dos digitos y acota a ANO_SANITY_RANGE."""
    s = s.copy()
    two_digit = s.between(0, 99)
    s = s.where(~two_digit, s + 2000)
    s = s.where(s.between(*ANO_SANITY_RANGE))
    return s


def derivar_discount_pct(df):
    """discount_pct por conjunto: licitaciones con-IVA, menores sin-IVA.

    NaN donde no es derivable (falta presupuesto, <= 0, o falta adjudicacion).
    """
    if "licitaciones" == df["conjunto"].iloc[0]:
        base, adj = df["importe_con_iva"], df["importe_adj_con_iva"]
    else:
        base, adj = df["importe_sin_iva"], df["importe_adjudicacion"]
    disc = (1 - adj / base) * 100
    return disc.where(base.notna() & (base > 0) & adj.notna())


def clasificar_estado(df):
    """Serie ml_estado ('train' | 'filtered' | NA=abierta) para un conjunto.

    Precedencia: fuera de ventana -> 'filtered' (recorte de poblacion);
    expediente abierto -> NA (aun sin objetivo, se clasificara cuando cierre);
    cerrado en ventana -> 'train' si algun objetivo es valido, si no 'filtered'.
    """
    es_lic = df["conjunto"].iloc[0] == "licitaciones"
    ano = fix_ano(df["ano"]).round().astype("Int64")

    num_ok = df["num_ofertas"].notna() & df["num_ofertas"].between(*NUM_OFERTAS_RANGE)
    disc = derivar_discount_pct(df)
    adj = df["importe_adj_con_iva"] if es_lic else df["importe_adjudicacion"]
    money_ok = disc.notna() & disc.between(*DISCOUNT_RANGE) & adj.notna() & (adj >= 0)

    en_ventana = ano.between(*WINDOW_YEARS).fillna(False).to_numpy(dtype=bool)
    if es_lic:
        # Cerrada = adjudicada/resuelta: solo entonces hay objetivo.
        cerrada = df["estado"].isin(AWARD_STATES).to_numpy(dtype=bool)
    else:
        # Sin filtro de estado (feed de adjudicaciones); abierta si aun no
        # consta ni recuento ni importe de adjudicacion.
        cerrada = (df["num_ofertas"].notna() | df["importe_adjudicacion"].notna()).to_numpy(dtype=bool)
    valida = (num_ok | money_ok).fillna(False).to_numpy(dtype=bool)

    estado = pd.Series(pd.NA, index=df.index, dtype="string")
    estado[~en_ventana] = "filtered"
    clasificable = en_ventana & cerrada
    estado[clasificable & valida] = "train"
    estado[clasificable & ~valida] = "filtered"
    return estado


def run(data_dir: Path, dry_run: bool = False) -> None:
    path = data_dir / "licitaciones.parquet"
    tabla = pq.read_table(path)
    df = tabla.select(COLS_NECESARIAS).to_pandas()

    # Clasificacion por conjunto (los criterios difieren).
    ml = pd.Series(pd.NA, index=df.index, dtype="string")
    for conjunto in ("licitaciones", "menores"):
        mask = df["conjunto"] == conjunto
        ml[mask] = clasificar_estado(df[mask])

    print(f"filas: {len(df):,}  ({path})")
    resumen = df.assign(ml_estado=ml).groupby(["conjunto", "ml_estado"], dropna=False).size()
    for (conjunto, estado), n in resumen.items():
        print(f"  {conjunto:12s} {str(estado):8s} {n:>10,}")

    # Diagnosticos de las filtered: fuera de ventana vs objetivo invalido.
    ano = fix_ano(df["ano"]).round().astype("Int64")
    en_ventana = ano.between(*WINDOW_YEARS).fillna(False)
    fuera = (ml == "filtered") & ~en_ventana
    invalido = (ml == "filtered") & en_ventana
    print(f"  filtered: fuera de ventana {int(fuera.sum()):,} | "
          f"objetivo invalido {int(invalido.sum()):,}")

    if dry_run:
        print("(dry-run: no se escribe)")
        return

    # Escritura atomica in place con el mismo esquema y compresion que scrape.py.
    idx = tabla.schema.get_field_index("ml_estado")
    col = pa.array(ml.astype("string"), type=pa.string())
    nueva = tabla.set_column(idx, "ml_estado", col)
    tmp = path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, tabla.schema, compression="snappy") as writer:
        for i in range(0, nueva.num_rows, 1_000_000):
            writer.write_table(nueva.slice(i, 1_000_000).cast(tabla.schema))
    os.replace(tmp, path)
    print(f"escrito: {path}  (ml_estado rellenado)")


def main() -> None:
    p = argparse.ArgumentParser(description="Clasificar ml_estado (train/filtered/abierta).")
    p.add_argument("--data-dir", default="Data", help="directorio con licitaciones.parquet")
    p.add_argument("--dry-run", action="store_true", help="informe sin escribir")
    args = p.parse_args()
    run(Path(args.data_dir), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
