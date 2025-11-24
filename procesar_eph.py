import pandas as pd
import numpy as np
import glob
import os

# ==========================
# CONFIG
# ==========================
CARPETA_EPH = "./"   # misma carpeta
ARCHIVO_IPC = "ipc_trimestral.csv"

COLUMNAS_NECESARIAS = {
    "P21": "Ingreso",
    "PONDIIO": "Ponderador",
    "REGION": "Region",
    "CH04": "Sexo",
    "CH06": "Edad",
    "ESTADO": "Estado"
}

REGIONES_VALIDAS = ["GBA", "Pampeana"]  # Buenos Aires + Córdoba

# ==========================
# FUNCIONES
# ==========================

def normalizar_columnas(df):
    cols = {}
    for c in df.columns:
        c_norm = c.strip().upper()
        cols[c] = c_norm
    df.rename(columns=cols, inplace=True)
    return df

def convertir_ingresos(df):
    if "P21" not in df.columns:
        print("⚠️  No se encontró columna P21 (ingresos).")
        return df

    # Reemplazar comas por puntos y convertir
    df["P21"] = (
        df["P21"]
        .astype(str)
        .str.replace(".", "", regex=False)     # eliminar separador miles
        .str.replace(",", ".", regex=False)    # convertir coma -> punto
    )

    df["P21"] = pd.to_numeric(df["P21"], errors="coerce")
    return df

def procesar_archivo(path):
    ext = path.split(".")[-1]

    if ext == "xls":
        df = pd.read_excel(path, engine="xlrd")
    else:
        df = pd.read_excel(path, engine="openpyxl")

    df = normalizar_columnas(df)
    df = convertir_ingresos(df)

    # Filtrar solo columnas relevantes
    disponibles = [c for c in COLUMNAS_NECESARIAS.keys() if c in df.columns]
    df = df[disponibles].copy()

    df.rename(columns=COLUMNAS_NECESARIAS, inplace=True)
    return df

# ==========================
# PROCESAMIENTO DE IPC
# ==========================

ipc = pd.read_csv(ARCHIVO_IPC)
ipc["ANO4"] = ipc["ANO4"].astype(int)
ipc["TRIMESTRE"] = ipc["TRIMESTRE"].astype(int)

# ==========================
# PROCESAR TODOS LOS EPH
# ==========================

archivos = sorted(glob.glob("usu_individual_*.*"))

if not archivos:
    print("❌ No se encontraron archivos EPH.")
    exit()

dfs = []

for arch in archivos:
    print(f"Procesando {arch}...")
    df = procesar_archivo(arch)

    # Extraer año y trimestre desde el nombre
    # ejemplo: usu_individual_t117  → t1 17
    nombre = os.path.basename(arch)
    partes = nombre.split("_")
    t = partes[-1].replace(".xlsx", "").replace(".xls", "").lower()

    # formato t117: t (trimestre=1), 17 (año=2017)
    if t.startswith("t") and len(t) >= 3:
        trim = int(t[1])
        anio = int("20" + t[2:4])
    else:
        raise ValueError(f"No pude interpretar año/trimestre desde {nombre}")

    df["ANO4"] = anio
    df["TRIMESTRE"] = trim

    dfs.append(df)

# Unir todo
eph = pd.concat(dfs, ignore_index=True)

# ==========================
# FILTRAR REGIONES
# ==========================

print("Filtrando regiones:", REGIONES_VALIDAS)
eph = eph[eph["Region"].isin(REGIONES_VALIDAS)]

print("Filas EPH luego de filtrar:", len(eph))

# ==========================
# MERGE CON IPC
# ==========================

eph = eph.merge(ipc, on=["ANO4", "TRIMESTRE", "REGION"], how="left")

print("Filas finales:", len(eph))
print("Muestra:")
print(eph.head())

# ==========================
# GUARDAR SALIDA
# ==========================

eph.to_csv("eph_final_con_ipc.csv", index=False, encoding="utf-8")

print("\n✔️ Archivo generado: eph_final_con_ipc.csv")
