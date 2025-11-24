import pandas as pd

# --- 1) Cargar archivo IPC ---
df = pd.read_csv("ipc_trimestral.csv", sep=";", encoding="latin1")

# --- 2) Asegurar que Periodo sea string ---
df["Periodo"] = df["Periodo"].astype(str)

# --- 3) Convertir Periodo (YYYYMM) → fecha ---
df["fecha"] = pd.to_datetime(df["Periodo"], format="%Y%m")

# --- 4) Nos quedamos solo con el Nivel General ---
df_nivel = df[df["Descripcion"] == "NIVEL GENERAL"]

# --- 5) Filtrar las regiones que necesitamos para Buenos Aires y Córdoba ---
regiones_validas = ["GBA", "Pampeana"]
df_nivel = df_nivel[df_nivel["Region"].isin(regiones_validas)]

print(df_nivel.head())
print(df_nivel["Region"].unique())
