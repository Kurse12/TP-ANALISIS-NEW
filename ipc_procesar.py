import pandas as pd

# Ruta a tu archivo IPC
ipc_path = "ipc_trimestral.csv"   # cámbialo si se llama distinto

# === 1. Cargar con separador ; si es csv, si es xlsx usar read_excel ===
if ipc_path.endswith(".csv"):
    ipc = pd.read_csv(ipc_path, sep=";", encoding="latin1")
else:
    ipc = pd.read_excel(ipc_path)

print("Columnas detectadas en IPC:")
print(ipc.columns)

# === 2. Crear Año, Mes y Trimestre ===
ipc['Periodo'] = ipc['Periodo'].astype(str)
ipc['ANO4'] = ipc['Periodo'].str[:4].astype(int)
ipc['MES'] = ipc['Periodo'].str[5:7].astype(int)
ipc['TRIMESTRE'] = ((ipc['MES'] - 1) // 3) + 1

# === 3. Renombrar columnas para unificarlas con la EPH ===
ipc.rename(columns={
    'Region': 'REGION',
    'Indice_IPC': 'IPC_VAL'
}, inplace=True)

# === 4. Quedarse con lo necesario ===
ipc_small = ipc[['ANO4', 'TRIMESTRE', 'REGION', 'IPC_VAL']].copy()

print(ipc_small.head())
print("IPC procesado correctamente. Tamaño:", len(ipc_small))
ipc_small.to_csv("ipc_trimestral.csv", index=False, encoding="utf-8")
print("Archivo guardado: ipc_trimestral.csv")

