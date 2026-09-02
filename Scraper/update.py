#!/usr/bin/env python3
"""
ACTUALIZACION INCREMENTAL DE LICITACIONES.PARQUET
=================================================
Refresca Data/licitaciones.parquet con los ultimos X meses (--meses,
default 12), reutilizando las primitivas de scrape.py:

  1. VENTANA   ZIPs cuyo periodo interseca la ventana (mes natural: mes en
               curso + X-1 anteriores): mensuales YYYYMM >= inicio y, si la
               ventana nace antes de lo mensual (2025), el anual del ano de
               inicio. Re-descarga FORZADA de esos ZIPs (delete + download):
               PLACSP los actualiza in-place y las adjudicaciones tardias de
               la ventana solo entran asi. El placeholder del mes en curso
               se tolera; al forzar siempre la ventana, el cache se
               auto-cura en el siguiente run.
  2. PARSE+CALIDAD  identico a scrape.py pero solo con las filas de la
               ventana. Los cuantiles por division CPV de INT-FIA-01/09 se
               calculan contra la ventana, no contra el historico completo
               (decision consensuada 2026-09-02): el score de estas filas
               puede diferir ligeramente del de un scrape completo.
  3. MERGE     contra el parquet existente, tres reglas (portadas del
               sistema anterior, spts/inference/store.py):
                 id nuevo        fila completa, columnas de inferencia vacias
                 id conocido     raw (41) + calidad (22) FRESCOS; preds,
                                 ml_estado y version se conservan
                 id ausente      fila intacta
               y una marca (protocolo del ciclo de evaluacion, ver
               Inference/evaluate.py): toda fila recien adjudicada — nueva,
               o refresco que cierra el expediente — que no sea train pasa a
               ml_estado='test': es el conjunto de evaluacion del sistema
               expuesto hasta el siguiente reentrenamiento.
  4. ESCRITURA atomica con el mismo FINAL_SCHEMA (escribir_parquet).

Mas alla de la marca 'test', no clasifica (la clasificacion plena es de
Modeling/cleaning.py, que pliega las test a train al reentrenar) ni
re-infiere preds (Inference/): las preds de una fila refrescada pueden
quedar desfasadas hasta la proxima inferencia.

Uso:
    python Scraper/update.py                  # ultimos 12 meses
    python Scraper/update.py --meses 24
    python Scraper/update.py --data-dir /otro/dir

Requiere Data/licitaciones.parquet (si no existe, scrape.py primero).
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import scrape  # mismo directorio: python Scraper/update.py

# ============================================================================
# VENTANA
# ============================================================================

def inicio_ventana(meses):
    """Primer mes natural de la ventana: mes en curso + (meses-1)
    anteriores (mes natural alineado: 12 meses = 12 ZIPs mensuales)."""
    hoy = datetime.now()
    return pd.Timestamp(hoy.year, hoy.month, 1) - pd.DateOffset(months=meses - 1)


def archivos_ventana(conjunto_id, meses):
    """Archivos (formato de generar_urls_conjunto) cuyo periodo interseca
    la ventana: mensuales YYYYMM >= inicio; anuales, solo el del ano de
    inicio si la ventana nace antes de lo mensual."""
    inicio = inicio_ventana(meses)
    ym_inicio = inicio.year * 100 + inicio.month
    hoy = datetime.now()
    candidatos = scrape.generar_urls_conjunto(conjunto_id, inicio.year, hoy.year)
    seleccion = []
    for archivo in candidatos:
        # patron_archivo: el periodo es el ultimo segmento "_..." del nombre
        # (4 digitos = anual, 6 = mensual)
        periodo = archivo["nombre"].rsplit("_", 1)[1].removesuffix(".zip")
        dentro = int(periodo) >= (ym_inicio if len(periodo) == 6 else inicio.year)
        if dentro:
            seleccion.append(archivo)
    return seleccion

# ============================================================================
# MERGE
# ============================================================================

ESTADOS_ADJUDICADOS = ["Resuelta", "Adjudicada"]  # espejo de cleaning.AWARD_STATES


def es_adjudicada(df, conjunto_id):
    """Condicion de expediente con objetivo disponible (espejo de la
    'cerrada' de Modeling/cleaning.py: licitaciones por estado, menores por
    recuento o importe de adjudicacion)."""
    if conjunto_id == "licitaciones":
        return df["estado"].isin(ESTADOS_ADJUDICADOS).fillna(False).to_numpy(dtype=bool)
    return (df["num_ofertas"].notna() | df["importe_adjudicacion"].notna()).to_numpy(dtype=bool)


def actualizar_conjunto(conjunto_id, zip_paths, tabla_vieja, borme, ted):
    """Parse + calidad de la ventana y merge con las filas previas del
    conjunto (las tres reglas). Devuelve (tabla final, estadisticas o None
    si la ventana no produjo registros). pandas solo con la ventana."""
    tabla_ventana = scrape.construir_tabla(zip_paths, conjunto_id)
    if tabla_ventana is None:
        print(f"[update] ⚠ {conjunto_id}: la ventana no produjo registros; "
              f"se conservan las {len(tabla_vieja):,} filas previas")
        return tabla_vieja, None

    # preds/estado/version a preservar salen del parquet viejo
    prev = (tabla_vieja.select(["id"] + scrape.PRED_COLS
                               + [scrape.ESTADO_COL, scrape.VERSION_COL])
            .to_pandas())

    df = tabla_ventana.to_pandas()
    del tabla_ventana
    df = scrape.aplicar_calidad(df, borme_path=borme,
                                ted_path=ted if conjunto_id == "licitaciones" else None)
    df = scrape.agregar_columnas_inferencia(df)
    df = scrape.preservar_inferencia_previa(df, prev)

    ids_viejos = set(prev["id"])
    en_ambos = df["id"].isin(ids_viejos)

    # de los refrescados, cuantos cambiaron de verdad (atom:updated)
    cols_adj = (["estado"] if conjunto_id == "licitaciones"
                else ["num_ofertas", "importe_adjudicacion"])
    viejo = (tabla_vieja.select(["id", "fecha_updated"] + cols_adj).to_pandas()
             .set_index("id"))
    v = viejo["fecha_updated"].reindex(df.loc[en_ambos, "id"]).reset_index(drop=True)
    n = df.loc[en_ambos, "fecha_updated"].reset_index(drop=True)
    cambio = ~((v == n) | (v.isna() & n.isna()))

    # Marca de test (protocolo del ciclo de evaluacion): recien adjudicada —
    # fila nueva, o conocida que antes no cumplia — que no sea train. El
    # resto de estados se preserva tal cual (las test existentes no se
    # desmarcan; cleaning las pliega a train al reentrenar).
    adj_ahora = es_adjudicada(df, conjunto_id)
    adj_antes = pd.Series(es_adjudicada(viejo, conjunto_id), index=viejo.index)
    recien = adj_ahora & ~adj_antes.reindex(df["id"]).fillna(False).to_numpy(dtype=bool)
    no_train = (df[scrape.ESTADO_COL] != "train").fillna(True).to_numpy(dtype=bool)
    marcar = recien & no_train
    df.loc[marcar, scrape.ESTADO_COL] = "test"

    tabla_nueva = pa.Table.from_pandas(df[scrape.FINAL_SCHEMA.names],
                                       schema=scrape.FINAL_SCHEMA,
                                       preserve_index=False)
    ids_ventana = set(df["id"])
    del df, prev

    # la ventana gana para los ids conocidos; el parquet viejo aporta solo
    # sus filas ausentes de la ventana
    mask = ~tabla_vieja.column("id").to_pandas().isin(ids_ventana).to_numpy()
    tabla_kept = tabla_vieja.take(np.flatnonzero(mask))
    del tabla_vieja
    tabla_final = pa.concat_tables([tabla_kept, tabla_nueva.cast(tabla_kept.schema)])

    stats = {"nuevos": int((~en_ambos).sum()), "refrescados": int(en_ambos.sum()),
             "cambiados": int(cambio.sum()), "test": int(marcar.sum()),
             "filas": len(tabla_final)}
    return tabla_final, stats

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Actualización incremental de licitaciones.parquet")
    parser.add_argument("--meses", type=int, default=12,
                        help="Meses de la ventana, alineada a mes natural: "
                             "mes en curso + X-1 anteriores (default: 12)")
    parser.add_argument("--data-dir", type=Path, default=scrape.DATA_DIR,
                        help=f"Árbol de datos (default: {scrape.DATA_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.meses < 1:
        raise SystemExit("✗ --meses debe ser >= 1")

    output_path = args.data_dir / "licitaciones.parquet"
    if not output_path.exists():
        raise SystemExit(f"✗ {output_path} no existe: ejecuta scrape.py primero")
    placsp_dir = args.data_dir / "downloads" / "placsp"
    refs_dir = args.data_dir / "references"
    inicio = inicio_ventana(args.meses)

    print("=" * 60)
    print("ACTUALIZACIÓN INCREMENTAL (PLACSP)")
    print("=" * 60)
    print(f"  Ventana: últimos {args.meses} meses (desde {inicio:%Y-%m}) | "
          f"Conjuntos: {', '.join(scrape.CONJUNTOS)}")
    print(f"  Datos: {args.data_dir}")

    # referencias de calidad (cacheadas; solo descargan si faltan)
    borme = scrape.asegurar_referencia("borme", refs_dir)
    ted = scrape.asegurar_referencia("ted", refs_dir)

    session = scrape.get_session()
    tablas = []
    for conjunto_id, config in scrape.CONJUNTOS.items():
        archivos = archivos_ventana(conjunto_id, args.meses)
        if archivos:
            print(f"\n[ventana] {config['nombre']}: {len(archivos)} ZIPs "
                  f"({archivos[0]['nombre']} .. {archivos[-1]['nombre']})")
        zip_paths = scrape.descargar_conjunto(
            session, conjunto_id, inicio.year, datetime.now().year,
            placsp_dir / conjunto_id, force=True, archivos=archivos)

        tabla_vieja = pq.read_table(output_path, filters=[("conjunto", "=", conjunto_id)])
        if not zip_paths:
            print(f"[update] ⚠ {conjunto_id}: sin ZIPs descargados; se conservan "
                  f"las {len(tabla_vieja):,} filas previas")
            tablas.append(tabla_vieja)
            continue

        tabla_final, stats = actualizar_conjunto(conjunto_id, zip_paths,
                                                 tabla_vieja, borme, ted)
        tablas.append(tabla_final)
        if stats:
            print(f"[merge] {conjunto_id}: {stats['nuevos']:,} nuevos | "
                  f"{stats['refrescados']:,} refrescados "
                  f"({stats['cambiados']:,} cambiaron) | "
                  f"{stats['test']:,} marcadas test | "
                  f"{stats['filas']:,} filas")

    if not tablas:
        raise SystemExit("✗ ningún conjunto disponible")
    scrape.escribir_parquet(tablas, output_path)
    print(f"\n✓ update completado: {sum(len(t) for t in tablas):,} licitaciones")


if __name__ == "__main__":
    main()
