#!/usr/bin/env python3
"""
SCRAPER DE LICITACIONES PUBLICAS (PLACSP)
=========================================
Crea Data/licitaciones.parquet con los conjuntos `licitaciones` (sin menores)
y `menores`, a partir de la sindicación ATOM de la Plataforma de Contratación
del Sector Público:

  1. DESCARGA   ZIPs anuales/mensuales -> Data/downloads/placsp/<conjunto>/
                (cache; --force re-descarga: PLACSP actualiza los ZIPs
                in-place y las adjudicaciones tardias solo entran con una
                re-descarga completa)
  2. PARSE      XML ATOM -> ~41 columnas raw (esquema congelado del sistema
                anterior, ver cabecera de COLUMNAS_RAW)
  3. CALIDAD    20 indicadores INT-* + score_calidad + es_menor (codigo de
                licitaciones-espana). Las referencias externas BORME y TED se
                descargan automaticamente del release de GitHub si faltan;
                si no se consiguen, INT-CONS-18/20 quedan a NaN pero las
                columnas existen (esquema estable)
  4. INFERENCIA columnas vacias que rellenan fases posteriores:
                5 predicciones + ml_estado + version. Si el parquet de
                salida ya existe, se preservan estas columnas de los ids
                conocidos (el raw y la calidad siempre se refrescan)

Notas de implementacion (lecciones del primer run completo):
  - los ZIPs se leen en streaming, sin extraer a disco (los anuales
    descomprimen varios GB y /tmp suele ser tmpfs)
  - las descargas son atomicas (.part + replace): un scrape interrumpido
    no deja un ZIP truncado que el cache tratara como valido
  - la acumulacion es en tablas Arrow por ZIP; pandas solo materializa un
    conjunto cada vez (menores son 2.75M filas y no cabe todo a la vez)
  - RAW_SCHEMA/FINAL_SCHEMA son explicitos: la inferencia de tipos falla
    cuando un ZIP viene con una columna entera a null (p.ej. es_pyme)

Decisiones consensuadas (2026-09-01): un solo parquet con ambos conjuntos
(columna `conjunto`), predicciones persistidas por fila, referencias con
descarga automatica, todo en este fichero, datos desde 2021. La columna de
ciclo de vida se llama ml_estado (no "estado": colisiona con el estado raw
del expediente y lo sobrescribiria).

Uso:
    python Scraper/scrape.py                      # 2021 .. año actual
    python Scraper/scrape.py --anos 2021-2026
    python Scraper/scrape.py --force              # re-descarga todos los ZIPs
    python Scraper/scrape.py --data-dir /otro/dir # árbol de datos alternativo

Origen del codigo: licitaciones-espana (nacional), a traves de la copia
vendor del sistema anterior, que ya incluia fixes (salida por conjunto,
escritura atomica, --force por actualizacion in-place de PLACSP).

La actualizacion incremental (ultimos X meses) vive en update.py; la
clasificacion training/filtered vive en Modeling/cleaning.py.
"""

import argparse
import json
import os
import re
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ============================================================================
# CONFIGURACION
# ============================================================================

# Arbol de datos por defecto: Data/ del repo (zips, referencias y parquet).
DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

# Conjuntos a scrapear (de los 5 que publica PLACSP, el sistema usa 2).
CONJUNTOS = {
    "licitaciones": {
        "nombre": "Licitaciones (sin menores)",
        "url_base": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_643/",
        "patron_archivo": "licitacionesPerfilesContratanteCompleto3_{periodo}.zip",
        "ano_inicio": 2012,
        "mensual_desde": 2025,  # 2025+ tiene archivos mensuales
    },
    "menores": {
        "nombre": "Contratos menores",
        "url_base": "https://contrataciondelsectorpublico.gob.es/sindicacion/sindicacion_1143/",
        "patron_archivo": "contratosMenoresPerfilesContratantes_{periodo}.zip",
        "ano_inicio": 2018,
        "mensual_desde": 2025,
    },
}

# Referencias externas de calidad: release de GitHub de licitaciones-espana.
REFERENCE_RELEASE = "https://github.com/BquantFinance/licitaciones-espana/releases/latest"
REFERENCE_FILES = {
    "borme": {  # INT-CONS-18: adjudicatario existe en el Registro Mercantil
        "asset": "borme.zip",
        "inner": "borme/data/borme_empresas_pub.parquet",
        "dest": "borme/borme_empresas_pub.parquet",
    },
    "ted": {  # INT-CONS-20: contrato SARA publicado en TED (solo licitaciones)
        "asset": "ted.zip",
        "inner": "ted/crossval_sara_v2.parquet",
        "dest": "ted/crossval_sara_v2.parquet",
    },
}

# Namespaces XML
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "cbc": "urn:dgpe:names:draft:codice:schema:xsd:CommonBasicComponents-2",
    "cac": "urn:dgpe:names:draft:codice:schema:xsd:CommonAggregateComponents-2",
    "cbc-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonBasicComponents-2",
    "cac-place-ext": "urn:dgpe:names:draft:codice-place-ext:schema:xsd:CommonAggregateComponents-2",
}

# Mapeos de codigos
TIPOS_CONTRATO = {
    "1": "Suministros", "2": "Servicios", "3": "Obras",
    "21": "Gestión Servicios Públicos", "31": "Concesión Obras",
    "40": "Concesión Servicios", "7": "Administrativo Especial",
    "8": "Privado", "50": "Patrimonial", "22": "22",
}

ESTADOS = {
    "PUB": "Publicada", "EV": "En evaluación", "ADJ": "Adjudicada",
    "RES": "Resuelta", "ANUL": "Anulada", "DES": "Desierta",
}

PROCEDIMIENTOS = {
    "1": "Abierto", "2": "Restringido", "3": "Negociado con publicidad",
    "4": "Negociado sin publicidad", "5": "Diálogo competitivo",
    "6": "Asociación innovación", "100": "Basado en acuerdo marco",
    "999": "Otros",
}

# Columnas de inferencia que scrape.py crea vacias y rellenan fases
# posteriores (training.py / inference.py / update.py).
PRED_COLS = [
    "num_ofertas_pred",        # float32, clip [0, 50]
    "zero_discount_prob",      # float32
    "zero_discount_pred",      # Int8, 1 = sin descuento
    "discount_pct_pred",       # float32, clip [0, 70]
    "system_discount_pct_pred" # float32, 0 donde el router dice sin descuento
]
# ml_estado (no "estado": colisiona con el estado del expediente, raw)
ESTADO_COL = "ml_estado"  # null (abierta) | test | train | filtered
VERSION_COL = "version"   # version del modelo que realizo la inferencia

# --- contratos de esquema Arrow ------------------------------------------------
# Explicitos: si un ZIP no trae ningun valor de una columna (p.ej. es_pyme
# todo None), la inferencia de tipos producira un null() incompatible con
# el resto de ZIPs al concatenar. Con esquema fijo, la concatenacion nunca
# depende del contenido. El orden es el de parsear_entry + conjunto + ano.
_COLS_RAW = [
    "conjunto", "id", "expediente", "objeto", "organo_contratante", "nif_organo",
    "dir3_organo", "id_plataforma", "ciudad_organo", "dependencia",
    "tipo_contrato_code", "tipo_contrato", "subtipo_code", "procedimiento_code",
    "procedimiento", "estado_code", "estado", "valor_estimado_contrato",
    "importe_sin_iva", "importe_con_iva", "importe_adjudicacion",
    "importe_adj_con_iva", "adjudicatario", "nif_adjudicatario", "num_ofertas",
    "es_pyme", "cpv_principal", "cpvs", "ubicacion", "nuts", "duracion",
    "duracion_unidad", "financiacion_ue", "urgencia", "fecha_limite",
    "hora_limite", "fecha_adjudicacion", "fecha_publicacion", "fecha_updated",
    "url", "ano",
]
_T_FLOAT = {"valor_estimado_contrato", "importe_sin_iva", "importe_con_iva",
            "importe_adjudicacion", "importe_adj_con_iva", "num_ofertas", "ano"}
_T_BOOL = {"es_pyme"}
_T_TS = {"fecha_limite", "fecha_adjudicacion", "fecha_publicacion"}

RAW_SCHEMA = pa.schema([
    (c, pa.float64() if c in _T_FLOAT
        else pa.bool_() if c in _T_BOOL
        else pa.timestamp("us", tz="UTC") if c == "fecha_updated"
        else pa.timestamp("us") if c in _T_TS
        else pa.string())
    for c in _COLS_RAW
])

# esquema del parquet de salida: raw + calidad + inferencia
_COLS_CALIDAD = ["INT-VAL-01", "INT-VAL-02", "INT-VAL-03", "INT-VAL-04", "INT-VAL-05",
                 "INT-VAL-06", "INT-VAL-07", "INT-VAL-09", "INT-VAL-10", "INT-VAL-12",
                 "INT-VAL-14", "INT-CONS-01", "INT-CONS-08", "INT-FIA-01", "INT-FIA-04",
                 "INT-FIA-08", "INT-FIA-09", "INT-FIA-11", "INT-CONS-20", "INT-CONS-18",
                 "score_calidad", "es_menor"]
FINAL_SCHEMA = pa.schema(
    list(RAW_SCHEMA)
    + [pa.field(c, pa.float64() if c == "score_calidad" else pa.bool_())
       for c in _COLS_CALIDAD]
    + [pa.field("num_ofertas_pred", pa.float32()),
       pa.field("zero_discount_prob", pa.float32()),
       pa.field("zero_discount_pred", pa.int8()),
       pa.field("discount_pct_pred", pa.float32()),
       pa.field("system_discount_pct_pred", pa.float32()),
       pa.field(ESTADO_COL, pa.string()),
       pa.field(VERSION_COL, pa.string())]
)

# ============================================================================
# DESCARGA
# ============================================================================

def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return session


def generar_urls_conjunto(conjunto_id, ano_inicio, ano_fin):
    """URLs de los archivos de un conjunto: anuales hasta 2024, mensuales
    desde 2025 (mes actual incluido; si el del mes en curso aun no existe,
    la descarga tolera el 404)."""
    config = CONJUNTOS[conjunto_id]
    ano_actual = datetime.now().year
    mes_actual = datetime.now().month
    ano_inicio = max(ano_inicio, config["ano_inicio"])

    archivos = []
    for ano in range(ano_inicio, ano_fin + 1):
        usar_mensual = config["mensual_desde"] is not None and ano >= config["mensual_desde"]
        if usar_mensual:
            max_mes = mes_actual if ano == ano_actual else 12
            periodos = [f"{ano}{mes:02d}" for mes in range(1, max_mes + 1)]
        else:
            periodos = [str(ano)]
        for periodo in periodos:
            nombre = config["patron_archivo"].format(periodo=periodo)
            archivos.append({"nombre": nombre, "url": config["url_base"] + nombre})
    return archivos


def descargar_archivo(session, url, filepath, max_reintentos=3):
    """Descarga un ZIP. Devuelve filepath, o None si no disponible/fallo."""
    if filepath.exists() and filepath.stat().st_size > 1000:
        print(f"   · ya existe ({filepath.stat().st_size / 1e6:.1f} MB)")
        return filepath

    for intento in range(max_reintentos):
        try:
            response = session.get(url, timeout=600, stream=True)
            response.raise_for_status()
            # .part + replace atomico: una descarga interrumpida no deja un
            # ZIP truncado que el cache trataria como valido
            tmp = filepath.with_suffix(filepath.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            tmp.replace(filepath)
            print(f"   ✓ {filepath.stat().st_size / 1e6:.1f} MB")
            return filepath
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("   ⚠ no disponible (404)")
                return None
            print(f"   ✗ error HTTP {e.response.status_code}")
        except Exception as e:
            print(f"   ✗ intento {intento + 1}/{max_reintentos}: {e}")
            time.sleep(2 ** intento)
    return None


def descargar_conjunto(session, conjunto_id, ano_inicio, ano_fin, zip_dir, force=False,
                       archivos=None):
    """Descarga los ZIPs de un conjunto. Devuelve las rutas obtenidas.
    `archivos` restringe a una lista pre-filtrada de generar_urls_conjunto
    (update.py: solo los ZIPs de la ventana)."""
    config = CONJUNTOS[conjunto_id]
    if archivos is None:
        archivos = generar_urls_conjunto(conjunto_id, ano_inicio, ano_fin)
    zip_dir.mkdir(parents=True, exist_ok=True)

    if force:
        # PLACSP actualiza los ZIPs in-place: sin esto, el skip-if-exists
        # ocultaria las adjudicaciones tardias.
        for archivo in archivos:
            filepath = zip_dir / archivo["nombre"]
            if filepath.exists():
                filepath.unlink()

    print(f"\n[descarga] {config['nombre']} — {len(archivos)} archivos")
    descargados = []
    for i, archivo in enumerate(archivos, 1):
        print(f"  [{i}/{len(archivos)}] {archivo['nombre']}", end="")
        filepath = zip_dir / archivo["nombre"]
        if descargar_archivo(session, archivo["url"], filepath):
            descargados.append(filepath)
        time.sleep(0.3)
    print(f"[descarga] {len(descargados)}/{len(archivos)} archivos disponibles")
    return descargados

# ============================================================================
# PARSE XML
# ============================================================================

def safe_text(element, xpath):
    if element is None:
        return None
    try:
        found = element.find(xpath, NS)
        if found is not None and found.text:
            return found.text.strip()
    except Exception:
        pass
    return None


def safe_attr(element, xpath, attr):
    if element is None:
        return None
    try:
        found = element.find(xpath, NS)
        if found is not None:
            return found.get(attr)
    except Exception:
        pass
    return None


def parsear_entry(entry):
    """Parsea una entrada atom:entry -> dict con las columnas raw."""
    try:
        id_lic = safe_text(entry, "atom:id")
        link = entry.find("atom:link", NS)
        url = link.get("href") if link is not None else None

        status = entry.find("cac-place-ext:ContractFolderStatus", NS)
        if status is None:
            return None

        expediente = safe_text(status, "cbc:ContractFolderID")
        estado_code = safe_text(status, "cbc-place-ext:ContractFolderStatusCode")

        located_party = status.find("cac-place-ext:LocatedContractingParty", NS)
        party = located_party.find("cac:Party", NS) if located_party is not None else None

        nombre_organo = safe_text(party, "cac:PartyName/cbc:Name")
        ciudad_organo = safe_text(party, "cac:PostalAddress/cbc:CityName")

        nif_organo = dir3_organo = id_plataforma = None
        if party is not None:
            for pid in party.findall("cac:PartyIdentification", NS):
                id_elem = pid.find("cbc:ID", NS)
                if id_elem is not None and id_elem.text:
                    scheme = id_elem.get("schemeName", "")
                    if scheme == "NIF":
                        nif_organo = id_elem.text.strip()
                    elif scheme == "DIR3":
                        dir3_organo = id_elem.text.strip()
                    elif scheme == "ID_PLATAFORMA":
                        id_plataforma = id_elem.text.strip()

        parent_names = []
        parent = (located_party.find("cac-place-ext:ParentLocatedParty", NS)
                  if located_party is not None else None)
        while parent is not None:
            pname = safe_text(parent, "cac:PartyName/cbc:Name")
            if pname:
                parent_names.append(pname)
            parent = parent.find("cac-place-ext:ParentLocatedParty", NS)
        dependencia = " > ".join(reversed(parent_names)) if parent_names else None

        project = status.find("cac:ProcurementProject", NS)
        objeto = safe_text(project, "cbc:Name")
        tipo_code = safe_text(project, "cbc:TypeCode")
        subtipo_code = safe_text(project, "cbc:SubTypeCode")

        valor_estimado_contrato = importe_sin_iva = importe_con_iva = None
        budget = project.find("cac:BudgetAmount", NS) if project is not None else None
        if budget is not None:
            for xpath, campo in [
                ("cbc:EstimatedOverallContractAmount", "valor_estimado_contrato"),
                ("cbc:TotalAmount", "importe_con_iva"),
                ("cbc:TaxExclusiveAmount", "importe_sin_iva"),
            ]:
                val = safe_text(budget, xpath)
                if val:
                    try:
                        if campo == "valor_estimado_contrato":
                            valor_estimado_contrato = float(val)
                        elif campo == "importe_con_iva":
                            importe_con_iva = float(val)
                        else:
                            importe_sin_iva = float(val)
                    except (ValueError, TypeError):
                        pass

        cpvs = []
        if project is not None:
            for cpv_elem in project.findall(".//cac:RequiredCommodityClassification/cbc:ItemClassificationCode", NS):
                if cpv_elem.text:
                    cpvs.append(cpv_elem.text.strip())
        cpv_principal = cpvs[0] if cpvs else None
        cpvs_todos = ";".join(cpvs) if cpvs else None

        ubicacion = safe_text(project, ".//cac:RealizedLocation/cbc:CountrySubentity")
        nuts = safe_text(project, ".//cac:RealizedLocation/cbc:CountrySubentityCode")

        duracion = safe_text(project, ".//cac:PlannedPeriod/cbc:DurationMeasure")
        duracion_unidad = safe_attr(project, ".//cac:PlannedPeriod/cbc:DurationMeasure", "unitCode")

        process = status.find("cac:TenderingProcess", NS)
        procedimiento_code = safe_text(process, "cbc:ProcedureCode")
        urgencia = safe_text(process, "cbc:UrgencyCode")

        fecha_limite = safe_text(process, ".//cac:TenderSubmissionDeadlinePeriod/cbc:EndDate")
        hora_limite = safe_text(process, ".//cac:TenderSubmissionDeadlinePeriod/cbc:EndTime")

        terms = status.find("cac:TenderingTerms", NS)
        financiacion_ue = safe_text(terms, "cbc:FundingProgramCode")

        result = status.find("cac:TenderResult", NS)
        adjudicatario = nif_adjudicatario = None
        importe_adjudicacion = importe_adj_con_iva = None
        fecha_adjudicacion = num_ofertas = es_pyme = None
        if result is not None:
            adjudicatario = safe_text(result, ".//cac:WinningParty/cac:PartyName/cbc:Name")

            winner_id = result.find(".//cac:WinningParty/cac:PartyIdentification/cbc:ID", NS)
            if winner_id is not None and winner_id.text:
                nif_adjudicatario = winner_id.text.strip()

            for xpath, campo in [
                (".//cac:AwardedTenderedProject/cac:LegalMonetaryTotal/cbc:TaxExclusiveAmount", "importe_adjudicacion"),
                (".//cac:AwardedTenderedProject/cac:LegalMonetaryTotal/cbc:PayableAmount", "importe_adj_con_iva"),
            ]:
                val = safe_text(result, xpath)
                if val:
                    try:
                        if campo == "importe_adjudicacion":
                            importe_adjudicacion = float(val)
                        else:
                            importe_adj_con_iva = float(val)
                    except (ValueError, TypeError):
                        pass

            fecha_adjudicacion = safe_text(result, "cbc:AwardDate")

            val = safe_text(result, "cbc:ReceivedTenderQuantity")
            if val:
                try:
                    num_ofertas = int(val)
                except (ValueError, TypeError):
                    pass

            pyme = safe_text(result, "cbc:SMEAwardedIndicator")
            es_pyme = pyme == "true" if pyme else None

        fecha_updated = safe_text(entry, "atom:updated")
        valid_notice = status.find("cac-place-ext:ValidNoticeInfo", NS)
        fecha_publicacion = safe_text(valid_notice, ".//cac-place-ext:AdditionalPublicationDocumentReference/cbc:IssueDate")

        return {
            "id": id_lic,
            "expediente": expediente,
            "objeto": objeto,
            "organo_contratante": nombre_organo,
            "nif_organo": nif_organo,
            "dir3_organo": dir3_organo,
            "id_plataforma": id_plataforma,
            "ciudad_organo": ciudad_organo,
            "dependencia": dependencia,
            "tipo_contrato_code": tipo_code,
            "tipo_contrato": TIPOS_CONTRATO.get(tipo_code, tipo_code),
            "subtipo_code": subtipo_code,
            "procedimiento_code": procedimiento_code,
            "procedimiento": PROCEDIMIENTOS.get(procedimiento_code, procedimiento_code),
            "estado_code": estado_code,
            "estado": ESTADOS.get(estado_code, estado_code),
            "valor_estimado_contrato": valor_estimado_contrato,
            "importe_sin_iva": importe_sin_iva,
            "importe_con_iva": importe_con_iva,
            "importe_adjudicacion": importe_adjudicacion,
            "importe_adj_con_iva": importe_adj_con_iva,
            "adjudicatario": adjudicatario,
            "nif_adjudicatario": nif_adjudicatario,
            "num_ofertas": num_ofertas,
            "es_pyme": es_pyme,
            "cpv_principal": cpv_principal,
            "cpvs": cpvs_todos,
            "ubicacion": ubicacion,
            "nuts": nuts,
            "duracion": duracion,
            "duracion_unidad": duracion_unidad,
            "financiacion_ue": financiacion_ue,
            "urgencia": urgencia,
            "fecha_limite": fecha_limite,
            "hora_limite": hora_limite,
            "fecha_adjudicacion": fecha_adjudicacion,
            "fecha_publicacion": fecha_publicacion,
            "fecha_updated": fecha_updated,
            "url": url,
        }
    except Exception:
        return None


def procesar_archivo_atom(source):
    """Procesa un archivo ATOM desde un objeto fichero (iterparse; fallback
    a parse completo)."""
    licitaciones = []
    try:
        context = ET.iterparse(source, events=("end",))
        for _event, elem in context:
            if elem.tag == "{http://www.w3.org/2005/Atom}entry":
                lic = parsear_entry(elem)
                if lic:
                    licitaciones.append(lic)
                elem.clear()
    except Exception:
        try:
            source.seek(0)
            root = ET.parse(source).getroot()
            for entry in root.findall("atom:entry", NS):
                lic = parsear_entry(entry)
                if lic:
                    licitaciones.append(lic)
        except Exception:
            pass
    return licitaciones


def procesar_zip(zip_path):
    """Parsea los .atom de un ZIP leyendolos en streaming. Sin extraccion a
    disco: los anuales descomprimen varios GB y /tmp suele ser tmpfs."""
    licitaciones = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            nombres = [n for n in zf.namelist() if n.endswith(".atom")]
            print(f"   {len(nombres)} atom", end=" ", flush=True)
            for nombre in nombres:
                with zf.open(nombre) as f:
                    licitaciones.extend(procesar_archivo_atom(f))
            print(f"-> {len(licitaciones):,} registros")
    except zipfile.BadZipFile:
        print("   ✗ ZIP corrupto")
    except Exception as e:
        print(f"   ✗ error: {e}")
    return licitaciones

# ============================================================================
# CALIDAD (codigo de licitaciones-espana, portado)
# ============================================================================

CALIDAD_CONFIG = {
    "importe_minimo": 1.0,
    "fecha_min": "1990-01-01",
    "fecha_max": "2030-12-31",
    "umbral_menor_obras": 40_000,
    "umbral_menor_servicios": 15_000,
    "umbral_menor_suministros": 15_000,
    "tolerancia_adj_lic": 0.05,
    "max_ofertas_global": 500,
    "min_plazo_dias": 0,
    "max_plazo_dias": 365,
    "pbl_outlier": 50_000_000,
}

CATALOGO = {
    "INT-VAL-01": "Importe de licitación en formato válido",
    "INT-VAL-02": "Importe de adjudicación en formato válido",
    "INT-VAL-03": "Importe mínimo plausible",
    "INT-VAL-04": "Número de licitadores es entero",
    "INT-VAL-05": "Número de licitadores no negativo",
    "INT-VAL-06": "Fecha de publicación en formato válido",
    "INT-VAL-07": "Fecha de adjudicación en formato válido",
    "INT-VAL-09": "Código CPV válido",
    "INT-VAL-10": "Código territorial válido",
    "INT-VAL-12": "Identificación válida del adjudicatario (NIF/NIE)",
    "INT-VAL-14": "Clasificación correcta procedimiento según cuantía",
    "INT-CONS-01": "Si hay adjudicación, num ofertas >= 1",
    "INT-CONS-08": "Importe coherente entre licitación y adjudicación",
    "INT-CONS-18": "Adjudicatario existe en BORME (Registro Mercantil)",
    "INT-CONS-20": "Contrato SARA publicado en TED",
    "INT-FIA-01": "Num ofertas dentro de rango razonable",
    "INT-FIA-04": "Plazo de presentación de ofertas razonable",
    "INT-FIA-08": "PBL atípico/inverosímil (outlier)",
    "INT-FIA-09": "PA plausible respecto a comparables por CPV",
    "INT-FIA-11": "Trazabilidad mínima del expediente",
}


def _dt(s):
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")


def _num(s):
    return s if pd.api.types.is_numeric_dtype(s) else pd.to_numeric(s, errors="coerce")


_NIF_LETRAS = "TRWAGMYFPDXBNJZSQVHLCKE"


def _nif_letra(n):
    return _NIF_LETRAS[int(n) % 23]


def _cif_ok(cif):
    t = cif[0]
    if t not in "ABCDEFGHJNPQRSUVW":
        return False
    d = cif[1:8]
    if not d.isdigit():
        return False
    sp = sum(int(x) for x in d[1::2])
    si = 0
    for x in d[0::2]:
        db = int(x) * 2
        si += db // 10 + db % 10
    ctrl = (10 - (sp + si) % 10) % 10
    cc = cif[8]
    if t in "KPQS":
        return cc == "JABCDEFGHI"[ctrl]
    elif t in "ABEH":
        return cc == str(ctrl)
    return cc == str(ctrl) or cc == "JABCDEFGHI"[ctrl]


def validar_nif(v):
    if pd.isna(v):
        return False
    raw = str(v)
    if "@" in raw:
        return False
    s = raw.strip().upper().replace("-", "").replace(" ", "").replace(".", "")
    if len(s) < 8 or len(s) > 9:
        return False
    if re.match(r"^\d{8}[A-Z]$", s):
        return s[8] == _nif_letra(s[:8])
    if re.match(r"^[XYZ]\d{7}[A-Z]$", s):
        return s[8] == _nif_letra({"X": "0", "Y": "1", "Z": "2"}[s[0]] + s[1:8])
    if re.match(r"^[A-Z]\d{7}[A-Z0-9]$", s):
        return _cif_ok(s)
    return False


_CPV_RE = re.compile(r"^\d{8}(-\d)?$")


def validar_cpv(v):
    if pd.isna(v):
        return False
    s = str(v).strip()
    # float64 desde parquet: "42933300.0" -> "42933300"
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    try:
        n = float(s.split("-")[0])
        if n == int(n):
            base = str(int(n)).zfill(8)
            if len(base) == 8:
                return True
    except (ValueError, OverflowError):
        pass
    return bool(_CPV_RE.match(s))


def div_cpv(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].replace("-", "").isdigit():
        s = s[:-2]
    try:
        n = float(s.split("-")[0])
        if n == int(n):
            s = str(int(n)).zfill(8)
    except (ValueError, OverflowError):
        pass
    return s[:2] if len(s) >= 2 and s[:2].isdigit() else ""


_NUTS_RE = re.compile(r"^ES[0-9A-Z]{0,3}$", re.I)


def validar_nuts(v):
    if pd.isna(v):
        return False
    return bool(_NUTS_RE.match(str(v).strip()))


def normalizar_nombre_empresa(nombre):
    if pd.isna(nombre):
        return ""
    s = str(nombre).upper().strip()
    for suf in [" SOCIEDAD LIMITADA", " SOCIEDAD ANONIMA", " S.L.U.", " S.L.L.",
                " S.L.", " S.A.U.", " S.A.", " SLU", " SLL", " SLP", " SL", " SAU", " SA",
                " S.COOP", " SCOOP", " S COOP", " S.C.", " SC", " UNIPERSONAL",
                " EN CONSTITUCION", ",", ".", "-"]:
        s = s.replace(suf, "")
    return re.sub(r"\s+", " ", s).strip()


def calcular_indicadores_base(df):
    """Los 17 indicadores sin dependencias externas."""
    r = pd.DataFrame(index=df.index)

    # VAL-01
    c = "importe_sin_iva" if "importe_sin_iva" in df.columns else "importe_con_iva" if "importe_con_iva" in df.columns else None
    r["INT-VAL-01"] = _num(df[c]).notna() if c else np.nan

    # VAL-02
    c = "importe_adjudicacion" if "importe_adjudicacion" in df.columns else "importe_adj_con_iva" if "importe_adj_con_iva" in df.columns else None
    r["INT-VAL-02"] = _num(df[c]).notna() if c else np.nan

    # VAL-03
    ic = [col for col in ["importe_sin_iva", "importe_con_iva", "importe_adjudicacion", "importe_adj_con_iva"] if col in df.columns]
    if ic:
        imp = df[ic].apply(pd.to_numeric, errors="coerce")
        r["INT-VAL-03"] = (imp >= CALIDAD_CONFIG["importe_minimo"]).any(axis=1) | imp.isna().all(axis=1)
    else:
        r["INT-VAL-03"] = np.nan

    # VAL-04/05
    if "num_ofertas" in df.columns:
        num = _num(df["num_ofertas"])
        r["INT-VAL-04"] = num.isna() | (num % 1 == 0)
        r["INT-VAL-05"] = num.isna() | (num >= 0)
    else:
        r["INT-VAL-04"] = np.nan
        r["INT-VAL-05"] = np.nan

    # VAL-06
    if "fecha_publicacion" in df.columns:
        fp = _dt(df["fecha_publicacion"])
        r["INT-VAL-06"] = fp.notna() & (fp >= CALIDAD_CONFIG["fecha_min"]) & (fp <= CALIDAD_CONFIG["fecha_max"])
    else:
        r["INT-VAL-06"] = np.nan

    # VAL-07
    if "fecha_adjudicacion" in df.columns:
        fa = _dt(df["fecha_adjudicacion"])
        r["INT-VAL-07"] = fa.notna() & (fa >= CALIDAD_CONFIG["fecha_min"]) & (fa <= CALIDAD_CONFIG["fecha_max"])
    else:
        r["INT-VAL-07"] = np.nan

    # VAL-09
    r["INT-VAL-09"] = df["cpv_principal"].apply(validar_cpv) if "cpv_principal" in df.columns else np.nan

    # VAL-10
    c = "nuts" if "nuts" in df.columns else "ubicacion" if "ubicacion" in df.columns else None
    r["INT-VAL-10"] = df[c].apply(validar_nuts) if c else np.nan

    # VAL-12
    r["INT-VAL-12"] = df["nif_adjudicatario"].apply(validar_nif) if "nif_adjudicatario" in df.columns else np.nan

    # VAL-14
    if "tipo_contrato" in df.columns:
        ci = "importe_adjudicacion" if "importe_adjudicacion" in df.columns else "importe_sin_iva" if "importe_sin_iva" in df.columns else None
        if ci:
            imp = _num(df[ci])
            tipo = df["tipo_contrato"].astype(str).str.lower().str.strip()
            if "conjunto" in df.columns:
                es_menor = df["conjunto"].astype(str).str.lower() == "menores"
                if "procedimiento" in df.columns:
                    es_menor = es_menor | df["procedimiento"].astype(str).str.lower().str.contains("menor", na=False)
            elif "procedimiento" in df.columns:
                es_menor = df["procedimiento"].astype(str).str.lower().str.contains("menor", na=False)
            else:
                es_menor = pd.Series(False, index=df.index)
            umbral = pd.Series(CALIDAD_CONFIG["umbral_menor_servicios"], index=df.index, dtype=float)
            umbral[tipo.str.contains("obra", na=False)] = CALIDAD_CONFIG["umbral_menor_obras"]
            umbral[tipo.str.contains("suministro", na=False)] = CALIDAD_CONFIG["umbral_menor_suministros"]
            r["INT-VAL-14"] = ~es_menor | imp.isna() | (imp <= umbral)
        else:
            r["INT-VAL-14"] = np.nan
    else:
        r["INT-VAL-14"] = np.nan

    # CONS-01
    if all(col in df.columns for col in ["estado", "num_ofertas"]):
        est = df["estado"].astype(str).str.lower().str.strip()
        r["INT-CONS-01"] = ~est.str.contains("adjud|formaliz|resuel", na=False) | (_num(df["num_ofertas"]) >= 1)
    else:
        r["INT-CONS-01"] = np.nan

    # CONS-08
    done = False
    for cl, ca in [("importe_sin_iva", "importe_adjudicacion"), ("importe_con_iva", "importe_adj_con_iva")]:
        if cl in df.columns and ca in df.columns:
            lic = _num(df[cl])
            adj = _num(df[ca])
            both = lic.notna() & adj.notna() & (lic > 0)
            r["INT-CONS-08"] = ~both | (adj <= lic * (1 + CALIDAD_CONFIG["tolerancia_adj_lic"]))
            done = True
            break
    if not done:
        r["INT-CONS-08"] = np.nan

    # FIA-01
    if "num_ofertas" in df.columns:
        num = _num(df["num_ofertas"])
        mx = CALIDAD_CONFIG["max_ofertas_global"]
        p = num.isna() | ((num >= 0) & (num <= mx))
        if "cpv_principal" in df.columns:
            dv = df["cpv_principal"].apply(div_cpv)
            p99 = num.groupby(dv).transform(lambda x: x.quantile(0.99)).fillna(mx)
            p = p & (num.isna() | (num <= p99))
        r["INT-FIA-01"] = p
    else:
        r["INT-FIA-01"] = np.nan

    # FIA-04
    if all(col in df.columns for col in ["fecha_publicacion", "fecha_limite"]):
        fp = _dt(df["fecha_publicacion"])
        fl = _dt(df["fecha_limite"])
        dias = (fl - fp).dt.days
        both = fp.notna() & fl.notna()
        r["INT-FIA-04"] = ~both | ((dias >= CALIDAD_CONFIG["min_plazo_dias"]) & (dias <= CALIDAD_CONFIG["max_plazo_dias"]))
    else:
        r["INT-FIA-04"] = np.nan

    # FIA-08
    c = "importe_sin_iva" if "importe_sin_iva" in df.columns else "importe_con_iva" if "importe_con_iva" in df.columns else None
    r["INT-FIA-08"] = (_num(df[c]).isna() | (_num(df[c]) <= CALIDAD_CONFIG["pbl_outlier"])) if c else np.nan

    # FIA-09
    ca = "importe_adjudicacion" if "importe_adjudicacion" in df.columns else "importe_adj_con_iva" if "importe_adj_con_iva" in df.columns else None
    if ca and "cpv_principal" in df.columns:
        pa = _num(df[ca])
        dv = df["cpv_principal"].apply(div_cpv)
        q1 = pa.groupby(dv).transform(lambda x: x.quantile(0.01))
        q99 = pa.groupby(dv).transform(lambda x: x.quantile(0.99))
        ev = pa.notna() & q1.notna() & q99.notna() & (pa > 0)
        r["INT-FIA-09"] = ~ev | ((pa >= q1) & (pa <= q99))
    else:
        r["INT-FIA-09"] = np.nan

    # FIA-11
    has_exp = df["expediente"].notna() if "expediente" in df.columns else pd.Series(False, index=df.index)
    has_url = df["url"].notna() if "url" in df.columns else pd.Series(False, index=df.index)
    has_id = df["id"].notna() if "id" in df.columns else pd.Series(False, index=df.index)
    r["INT-FIA-11"] = has_exp & (has_url | has_id)

    return r


def calcular_cons20(df, path_ted):
    """INT-CONS-20: cruce con el crosswalk TED de contratos SARA."""
    print(f"  [calidad] cargando TED: {path_ted}")
    ted = pd.read_parquet(path_ted, columns=["expediente", "nif_adjudicatario",
                                             "_ted_validated", "_ted_missing",
                                             "_match_strategy"])
    print(f"  [calidad] {len(ted):,} contratos SARA")
    ted["_key"] = ted["expediente"].astype(str) + "|" + ted["nif_adjudicatario"].astype(str)
    td = dict(zip(ted["_key"], ted["_ted_validated"]))
    n_val = ted["_ted_validated"].sum()
    print(f"  [calidad] validados por 5 estrategias: {n_val:,} ({n_val / len(ted) * 100:.1f}%)")
    del ted
    if all(col in df.columns for col in ["expediente", "nif_adjudicatario"]):
        keys = df["expediente"].astype(str) + "|" + df["nif_adjudicatario"].astype(str)
        res = keys.map(td)
        n_eval = res.notna().sum()
        n_ok = (res == True).sum()  # noqa: E712 — comparacion contra la columna booleana del crosswalk
        print(f"  [calidad] SARA en nacional: {n_eval:,} | en TED: {n_ok:,} ({n_ok / max(n_eval, 1) * 100:.1f}%)")
        return res
    return pd.Series(np.nan, index=df.index)


def cargar_borme(path):
    print(f"  [calidad] cargando BORME: {path}")
    b = pd.read_parquet(path, columns=["empresa_norm"])
    empresas = set(b["empresa_norm"].dropna().unique())
    del b
    print(f"  [calidad] {len(empresas):,} empresas únicas")
    return empresas


def aplicar_borme(df, empresas):
    """INT-CONS-18: adjudicatarios sociedad presentes en BORME."""
    if empresas is None or "adjudicatario" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=object).astype("boolean")
    adj_n = df["adjudicatario"].apply(normalizar_nombre_empresa)
    tiene = df["adjudicatario"].notna() & (adj_n != "")
    es_emp = (df["nif_adjudicatario"].astype(str).str.strip().str.upper()
              .str.match(r"^[A-HJ-NP-SUVW]", na=False)
              if "nif_adjudicatario" in df.columns else pd.Series(False, index=df.index))
    ev = tiene & es_emp
    if ev.sum() == 0:
        return pd.Series(np.nan, index=df.index, dtype=object).astype("boolean")
    enc = adj_n.isin(empresas)
    res = pd.Series(np.nan, index=df.index, dtype=object)
    res.loc[ev] = enc.loc[ev]
    return res.astype("boolean")


def calcular_score(sc):
    ev = sc.notna().sum(axis=1)
    pa = sc.fillna(False).sum(axis=1)
    s = (pa / ev * 100).round(1)
    s[ev == 0] = np.nan
    return s


def imprimir_resumen_calidad(sc, df):
    """Incumplimiento por indicador (>5% marcado) y menores vs regulares."""
    print(f"\n[calidad] resumen ({len(df):,} contratos)")
    for col in sc.columns:
        s = sc[col]
        t = s.notna().sum()
        if t == 0:
            continue
        inv = int(t - s.sum())
        pct = inv / t * 100
        marca = "!!" if pct > 5 else "  "
        print(f"  {marca}{col:<14s} | {pct:5.1f}% | {t:>10,} eval | {CATALOGO.get(col, '')[:50]}")

    if "conjunto" in df.columns:
        conj = df["conjunto"].astype(str).str.lower()
        es_men = conj == "menores"
        nm = es_men.sum()
        if 0 < nm < len(df):
            print(f"\n  -- menores ({nm:,}) vs regulares ({len(df) - nm:,}) --")
            for col in sc.columns:
                sm = sc.loc[es_men, col]
                sr = sc.loc[~es_men, col]
                tm, tr = sm.notna().sum(), sr.notna().sum()
                if tm == 0 or tr == 0:
                    continue
                pm = (tm - sm.sum()) / tm * 100
                pr = (tr - sr.sum()) / tr * 100
                flag = " <<<" if abs(pm - pr) > 10 else ""
                print(f"  {col:<16s} | men {pm:6.1f}% | reg {pr:6.1f}% | diff {pm - pr:+7.1f}pp{flag}")


def aplicar_calidad(df, borme_path=None, ted_path=None):
    """Añade al dataframe: 17 INT-* base + INT-CONS-18/20 + score_calidad
    + es_menor. Las columnas externas existen siempre; a NaN si falta la
    referencia (esquema estable)."""
    sc = calcular_indicadores_base(df)

    if ted_path:
        print("  [calidad] INT-CONS-20 (TED SARA)...")
        sc["INT-CONS-20"] = calcular_cons20(df, ted_path)
    else:
        sc["INT-CONS-20"] = pd.Series(np.nan, index=df.index, dtype=object)

    if borme_path:
        print("  [calidad] INT-CONS-18 (BORME)...")
        r18 = aplicar_borme(df, cargar_borme(borme_path))
        if r18.notna().any():
            sc["INT-CONS-18"] = r18
        else:
            sc["INT-CONS-18"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    else:
        sc["INT-CONS-18"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    score = calcular_score(sc)
    imprimir_resumen_calidad(sc, df)

    # asignacion in-place: una concat copiaria el DataFrame entero (GB en
    # menores) solo para añadir columnas booleanas
    for col in sc.columns:
        df[col] = sc[col]
    df["score_calidad"] = score
    df["es_menor"] = df["conjunto"].astype(str).str.lower().values == "menores"
    return df

# ============================================================================
# REFERENCIAS EXTERNAS (release de GitHub de licitaciones-espana)
# ============================================================================

def _release_assets():
    """asset name -> browser_download_url del latest release configurado."""
    url = REFERENCE_RELEASE.rstrip("/")
    parts = url.split("/")
    if len(parts) < 7 or parts[2] != "github.com" or parts[-2:] != ["releases", "latest"]:
        raise ValueError(f"URL de release inesperada: {url!r}")
    owner, repo = parts[-4], parts[-3]
    api = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    req = urllib.request.Request(api, headers={"User-Agent": "nuevo-licitaciones"})
    with urllib.request.urlopen(req, timeout=60) as r:
        release = json.load(r)
    return {a["name"]: a["browser_download_url"] for a in release["assets"]}


def _descargar(url, dest):
    """Descarga en streaming a .part y reemplazo atomico."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nuevo-licitaciones"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)


def asegurar_referencia(key, refs_dir):
    """Devuelve la ruta del parquet de referencia, descargandolo del release
    si falta (zip ~0.2-0.8 GB; se extrae solo el parquet interno y se borra
    el zip). None si no se pudo conseguir — calidad seguira sin ese join."""
    spec = REFERENCE_FILES[key]
    dest = refs_dir / spec["dest"]
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"[referencias] {key}: presente ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    try:
        assets = _release_assets()
        if spec["asset"] not in assets:
            raise KeyError(f"asset {spec['asset']!r} no esta en el latest release "
                           f"(tiene: {sorted(assets)})")
        print(f"[referencias] {key}: descargando {spec['asset']} ...")
        zip_path = refs_dir / spec["asset"]
        _descargar(assets[spec["asset"]], zip_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            # el parquet interno puede ir precedido de carpetas; match por sufijo
            miembro = next(m for m in zf.namelist()
                           if m == spec["inner"] or m.endswith("/" + spec["inner"])
                           or m.endswith(spec["inner"].split("/")[-1]))
            with zf.open(miembro) as src, open(dest, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
        zip_path.unlink()  # el parquet interno es el artefacto que se cachea
        print(f"[referencias] {key}: {dest.stat().st_size / 1e6:.0f} MB -> {dest}")
        return dest
    except Exception as e:
        print(f"[referencias] ⚠ {key}: no disponible ({e}); "
              f"INT-CONS-{18 if key == 'borme' else 20} quedara a NaN")
        return None

# ============================================================================
# ENSAMBLADO
# ============================================================================

def construir_tabla(zip_paths, conjunto_id):
    """Parsea los ZIPs de un conjunto -> tabla Arrow con el esquema raw,
    columna conjunto/ano y dedupe global keep-last por id.

    Memoria: cada ZIP se convierte a tabla Arrow (columnar, compacta) y el
    pandas se libera antes del siguiente — acumular los DataFrames pandas
    (o los dicts) de todos los ZIPs no cabe en RAM en menores (2.75M filas).
    El dedupe es global (cronologico: el ZIP mas reciente gana), igual que
    en el sistema anterior.
    """
    tablas_zip = []
    for i, zip_path in enumerate(zip_paths, 1):
        print(f"  [parse] [{i}/{len(zip_paths)}] {zip_path.name}", end="")
        registros = procesar_zip(zip_path)
        if not registros:
            continue
        df = pd.DataFrame(registros)
        del registros
        df.insert(0, "conjunto", conjunto_id)

        for col in ("fecha_limite", "fecha_adjudicacion", "fecha_publicacion"):
            df[col] = pd.to_datetime(df[col], errors="coerce").astype("datetime64[us]")
        df["fecha_updated"] = (pd.to_datetime(df["fecha_updated"], errors="coerce", utc=True)
                               .astype("datetime64[us, UTC]"))
        df["ano"] = df["fecha_publicacion"].dt.year.astype("float64")  # NaN si sin fecha
        for col in ("valor_estimado_contrato", "importe_sin_iva", "importe_con_iva",
                    "importe_adjudicacion", "importe_adj_con_iva", "num_ofertas"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        n_zip = len(df)
        df = df.drop_duplicates(subset=["id"], keep="last")
        if n_zip != len(df):
            print(f"   (dentro del ZIP: {n_zip - len(df):,} duplicados)")
        tablas_zip.append(pa.Table.from_pandas(df[RAW_SCHEMA.names], schema=RAW_SCHEMA,
                                               preserve_index=False))
        del df

    if not tablas_zip:
        return None
    tabla = pa.concat_tables(tablas_zip) if len(tablas_zip) > 1 else tablas_zip[0]
    del tablas_zip

    # dedupe global keep-last (los chunks estan en orden cronologico)
    ids = tabla.column("id").to_pandas()
    duplicados = ids.duplicated(keep="last")
    if duplicados.any():
        print(f"  [parse] duplicados entre ZIPs: {int(duplicados.sum()):,}")
        tabla = tabla.take(np.flatnonzero(~duplicados.to_numpy()))
    return tabla


def agregar_columnas_inferencia(df):
    """Crea vacias las columnas que rellenan las fases posteriores."""
    df["num_ofertas_pred"] = pd.Series(np.nan, index=df.index, dtype="float32")
    df["zero_discount_prob"] = pd.Series(np.nan, index=df.index, dtype="float32")
    df["zero_discount_pred"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df["discount_pct_pred"] = pd.Series(np.nan, index=df.index, dtype="float32")
    df["system_discount_pct_pred"] = pd.Series(np.nan, index=df.index, dtype="float32")
    df[ESTADO_COL] = pd.Series(None, index=df.index, dtype="object")
    df[VERSION_COL] = pd.Series(None, index=df.index, dtype="object")
    return df


def cargar_inferencia_previa(output_path):
    """Lee [id + predicciones/estado/version] del parquet de salida previo
    (si existe) para preservarlos en el re-scrape."""
    cols = ["id"] + PRED_COLS + [ESTADO_COL, VERSION_COL]
    if not output_path.exists():
        return None
    prev = pd.read_parquet(output_path, columns=cols)
    prev = prev.drop_duplicates(subset=["id"], keep="last")
    return prev


def preservar_inferencia_previa(df, prev):
    """Conserva predicciones/estado/version de los ids conocidos (asignacion
    in-place: un merge copiaria el DataFrame entero). Raw y calidad siempre
    frescos."""
    cols = PRED_COLS + [ESTADO_COL, VERSION_COL]
    prev_idx = prev.set_index("id")
    mask = df["id"].isin(prev_idx.index)
    if mask.any():
        # id repetido no puede ocurrir: df viene del dedupe por id
        prev_vals = prev_idx.loc[df.loc[mask, "id"], cols].reset_index(drop=True)
        # asignacion posicional (iloc): loc alinearia por etiquetas de indice
        filas = np.flatnonzero(mask.to_numpy())
        df.iloc[filas, [df.columns.get_loc(c) for c in cols]] = prev_vals
    print(f"[salida] {int(mask.sum()):,}/{len(df):,} ids previos: "
          f"inferencia/estado preservados")
    return df


def escribir_parquet(tablas, output_path):
    """Ensambla los Arrow tables de cada conjunto en el parquet final.
    Cada conjunto se mantiene como tabla Arrow (columnar, compacta): los
    dos DataFrames pandas a la vez no caben en RAM. Escritura atomica."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    schema = tablas[0].schema
    with pq.ParquetWriter(tmp, schema, compression="snappy") as writer:
        for tabla in tablas:
            writer.write_table(tabla.cast(schema))
    os.replace(tmp, output_path)
    n_filas = sum(len(t) for t in tablas)
    print(f"[salida] {output_path} — {n_filas:,} filas x {len(schema.names)} columnas "
          f"({output_path.stat().st_size / 1e6:.0f} MB)")


def resumen_conjunto(df):
    anos = df["ano"].dropna()
    span = f"{anos.min():.0f}-{anos.max():.0f}" if len(anos) else "s/f"
    adjudicadas = df["estado"].str.contains("Adjudicada|Resuelta", na=False).mean() * 100
    importe = df["importe_sin_iva"].sum() / 1e9
    score = df["score_calidad"].mean()
    print(f"[resumen] {df['conjunto'].iloc[0]:<14s} {len(df):>10,} filas | {span} | "
          f"adj {adjudicadas:4.1f}% | {importe:6.2f}B € | score {score:.1f}")

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Scraper de licitaciones públicas (PLACSP)")
    parser.add_argument("--anos", type=str, default=f"2021-{date.today().year}",
                        help="Rango de años (default: 2021-<año actual>)")
    parser.add_argument("--force", action="store_true",
                        help="Re-descargar todos los ZIPs (PLACSP actualiza in-place; "
                             "las adjudicaciones tardías solo entran así)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"Árbol de datos (default: {DATA_DIR})")
    return parser.parse_args()


def main():
    args = parse_args()

    partes = args.anos.split("-")
    ano_inicio = int(partes[0])
    ano_fin = int(partes[1]) if len(partes) > 1 else ano_inicio

    placsp_dir = args.data_dir / "downloads" / "placsp"
    refs_dir = args.data_dir / "references"
    output_path = args.data_dir / "licitaciones.parquet"

    print("=" * 60)
    print("SCRAPER DE LICITACIONES PÚBLICAS (PLACSP)")
    print("=" * 60)
    print(f"  Años: {ano_inicio}-{ano_fin} | Conjuntos: {', '.join(CONJUNTOS)}"
          f"{' | force' if args.force else ''}")
    print(f"  Datos: {args.data_dir}")

    # referencias de calidad (descarga automatica si faltan)
    borme = asegurar_referencia("borme", refs_dir)
    ted = asegurar_referencia("ted", refs_dir)

    session = get_session()
    prev = cargar_inferencia_previa(output_path)
    tablas = []
    for conjunto_id, config in CONJUNTOS.items():
        zip_paths = descargar_conjunto(session, conjunto_id, ano_inicio, ano_fin,
                                       placsp_dir / conjunto_id, force=args.force)
        if not zip_paths:
            print(f"[scrape] ⚠ {conjunto_id}: sin archivos, se omite")
            continue

        print(f"\n[parse] {config['nombre']}")
        tabla = construir_tabla(zip_paths, conjunto_id)
        if tabla is None:
            print(f"[scrape] ⚠ {conjunto_id}: sin registros, se omite")
            continue

        # pandas solo aqui, un conjunto cada vez (la tabla Arrow queda
        # libre al convertir); es lo que RAM permite
        df = tabla.to_pandas()
        del tabla

        # TED solo aplica a licitaciones (contratos SARA); BORME a ambos
        df = aplicar_calidad(df, borme_path=borme,
                             ted_path=ted if conjunto_id == "licitaciones" else None)
        df = agregar_columnas_inferencia(df)
        if prev is not None:
            df = preservar_inferencia_previa(df, prev)
        resumen_conjunto(df)

        tablas.append(pa.Table.from_pandas(df[FINAL_SCHEMA.names], schema=FINAL_SCHEMA,
                                           preserve_index=False))
        del df

    if not tablas:
        raise SystemExit("✗ ningún conjunto produjo datos")

    escribir_parquet(tablas, output_path)
    print(f"\n✓ scrape completado: {sum(len(t) for t in tablas):,} licitaciones")


if __name__ == "__main__":
    main()
