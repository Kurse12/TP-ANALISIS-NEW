import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# ================= CONFIG =================
IPC_FILE = "ipc_trimestral.csv"
EPH_PATTERN = "eph/raw/usu_*t*.xls*"
PROCESSED_DIR = "eph/processed"
GRAF_DIR = "graficos"

REGIONES_MAP = {
    1: "GBA",
    2: "Pampeana"
}

# ================= FUNCIONES =================
def cargar_ipc(path):
    ipc = pd.read_csv(path)
    ipc.columns = ipc.columns.str.strip().str.upper()
    ipc["ANO4"] = ipc["ANO4"].astype(int)
    ipc["TRIMESTRE"] = ipc["TRIMESTRE"].astype(int)
    ipc["REGION"] = ipc["REGION"].astype(str)
    ipc["IPC_VAL"] = ipc["IPC_VAL"].astype(str).str.replace(",", ".").astype(float)
    ipc = ipc[ipc["REGION"].isin(REGIONES_MAP.values())].copy()
    return ipc

def cargar_eph(pattern):
    archivos = glob.glob(pattern)
    if not archivos:
        raise ValueError("No se encontraron archivos EPH.")
    dfs = []
    for file in archivos:
        print(f"Cargando {file}")
        try:
            df = pd.read_excel(file, engine="openpyxl")
        except:
            df = pd.read_excel(file)
        df.columns = df.columns.str.upper().str.strip()
        column_map = {
            "TRIMESTRE": "TRIMESTRE", "TRIMES": "TRIMESTRE", "T_TRIM": "TRIMESTRE",
            "TRIM": "TRIMESTRE", "NROTRIM": "TRIMESTRE",
            "ANO4": "ANO4", "ANIO4": "ANO4", "AÑO4": "ANO4"
        }
        for k, v in column_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        df["TRIMESTRE"] = df["TRIMESTRE"].astype(str).str.extract(r"(\d)").astype(int)
        df["ANO4"] = df["ANO4"].astype(str).str.extract(r"(\d{4})").astype(int)
        dfs.append(df)
    eph = pd.concat(dfs, ignore_index=True)
    return eph

def filtrar_regiones(eph):
    eph = eph[eph["REGION"].isin(REGIONES_MAP.keys())].copy()
    eph["REGION_NAME"] = eph["REGION"].map(REGIONES_MAP)
    return eph

def calcular_tasas(eph):
    eph["OCUPADO"] = (eph["ESTADO"] == 0).astype(int)
    eph["DESOCUPADO"] = (eph["ESTADO"] == 1).astype(int)
    eph["ACTIVO"] = eph["OCUPADO"] | eph["DESOCUPADO"]
    tasas = eph.groupby(["ANO4", "TRIMESTRE", "REGION_NAME"]).agg(
        tasa_empleo=("OCUPADO", "mean"),
        tasa_desocupacion=("DESOCUPADO", "mean"),
        tasa_actividad=("ACTIVO", "mean")
    ).reset_index()
    return tasas

def ajustar_ingresos(eph, ipc):
    if "P21" not in eph.columns:
        raise ValueError("No existe la columna P21 en la EPH.")
    eph["P21"] = eph["P21"].astype(str).str.replace(",", ".", regex=False).str.extract(r"([\d\.]+)")
    eph["P21"] = pd.to_numeric(eph["P21"], errors="coerce").astype("float32")
    ipc["IPC_VAL"] = ipc["IPC_VAL"].astype("float32")
    resultados = []
    for (ano, trim, reg), df_chunk in eph.groupby(["ANO4", "TRIMESTRE", "REGION_NAME"]):
        ipc_val = ipc[(ipc["ANO4"]==ano)&(ipc["TRIMESTRE"]==trim)&
                      ((ipc["REGION"]=="GBA") if reg=="GBA" else (ipc["REGION"]=="Pampeana"))]["IPC_VAL"].values
        df_chunk["INGRESO_REAL"] = df_chunk["P21"] / (ipc_val[0] if len(ipc_val)>0 else np.nan)
        resultados.append(df_chunk)
    eph_adj = pd.concat(resultados, ignore_index=True)
    return eph_adj

def graficar_tasas(tasas):
    os.makedirs(GRAF_DIR, exist_ok=True)
    tasas = tasas.sort_values(["ANO4", "TRIMESTRE"])
    tasas["PERIODO"] = tasas["ANO4"].astype(str) + "T" + tasas["TRIMESTRE"].astype(str)
    for col in ["tasa_empleo", "tasa_desocupacion", "tasa_actividad"]:
        plt.figure(figsize=(12,5))
        for reg in tasas["REGION_NAME"].unique():
            df = tasas[tasas["REGION_NAME"]==reg]
            plt.plot(df["PERIODO"], df[col], marker="o", label=reg)
        plt.title(col.replace("_"," ").upper())
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{GRAF_DIR}/{col}.png")
        plt.close()

def graficar_ingreso_multivariado(eph_adj, variable="SEXO"):
    os.makedirs(GRAF_DIR, exist_ok=True)
    eph_grouped = eph_adj.groupby(["ANO4","TRIMESTRE","REGION_NAME",variable]).agg(
        ingreso_real_prom=("INGRESO_REAL","mean")
    ).reset_index()
    eph_grouped["PERIODO"] = eph_grouped["ANO4"].astype(str)+"T"+eph_grouped["TRIMESTRE"].astype(str)
    eph_grouped = eph_grouped.sort_values(["ANO4","TRIMESTRE"])
    plt.figure(figsize=(14,6))
    for reg in eph_grouped["REGION_NAME"].unique():
        for val in eph_grouped[variable].unique():
            df = eph_grouped[(eph_grouped["REGION_NAME"]==reg) & (eph_grouped[variable]==val)]
            plt.plot(df["PERIODO"], df["ingreso_real_prom"], marker="o", label=f"{reg}-{val}")
    plt.title(f"INGRESO REAL PROMEDIO POR {variable.upper()} Y REGIÓN", fontsize=16)
    plt.xlabel("Período", fontsize=12)
    plt.ylabel("Ingreso Real", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{GRAF_DIR}/ingreso_real_multivariado_{variable}.png")
    plt.close()

# ================= MAIN =================
def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("\n=== CARGANDO IPC ===")
    ipc = cargar_ipc(IPC_FILE)
    print("\n=== CARGANDO EPH ===")
    eph = cargar_eph(EPH_PATTERN)
    print("EPH cargada:", eph.shape)
    print("\n=== FILTRANDO REGIONES ===")
    eph = filtrar_regiones(eph)
    print("EPH post-filtrado:", eph.shape)
    print("\n=== CALCULANDO TASAS ===")
    tasas = calcular_tasas(eph)
    print("\n=== AJUSTANDO INGRESOS ===")
    eph_adj = ajustar_ingresos(eph, ipc)
    print("\n=== GUARDANDO PROCESADOS POR TRIMESTRE ===")
    for t in eph["TRIMESTRE"].unique():
        df_to_save = eph[eph["TRIMESTRE"]==t].copy()
        # Convertir fechas a string para evitar errores parquet
        for col in df_to_save.select_dtypes(include=["datetime64"]).columns:
            df_to_save[col] = df_to_save[col].astype(str)
        fname = f"{PROCESSED_DIR}/eph_t{t}.parquet"
        df_to_save.to_parquet(fname, index=False)
        print(f"Guardado: {fname}")
    print("\n=== GRAFICANDO TASAS ===")
    graficar_tasas(tasas)
    print("\n=== GRAFICANDO INGRESO MULTIVARIADO ===")
    graficar_ingreso_multivariado(eph_adj, variable="SEXO")  # Cambiar a NIVEL_EDUCATIVO si querés
    print("\n✔ Todo OK. Archivos procesados y gráficos generados.")

if __name__=="__main__":
    main()
