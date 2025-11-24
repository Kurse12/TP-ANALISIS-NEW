import pandas as pd

print("=" * 80)
print("LIMPIEZA DEL ARCHIVO IPC")
print("=" * 80)

# Cargar IPC original
ipc = pd.read_csv('ipc_trimestral.csv')
print(f"\n1. IPC original: {len(ipc):,} registros")

# Convertir IPC_VAL a numérico (reemplazar coma por punto)
ipc['IPC_VAL'] = ipc['IPC_VAL'].astype(str).str.replace(',', '.').astype(float)
print(f"✓ IPC_VAL convertido a numérico")

# Eliminar duplicados completos
ipc = ipc.drop_duplicates(subset=['ANO4', 'TRIMESTRE', 'REGION'], keep='first')
print(f"✓ Duplicados eliminados: {len(ipc):,} registros")

# Filtrar solo trimestres válidos (1-4, eliminar 0)
ipc = ipc[ipc['TRIMESTRE'].isin([1, 2, 3, 4])]
print(f"✓ Solo trimestres 1-4: {len(ipc):,} registros")

# Verificar resultado
print("\n2. RESUMEN DEL IPC LIMPIO:")
print(f"   Registros totales: {len(ipc):,}")
print(f"   Años: {ipc['ANO4'].min()} - {ipc['ANO4'].max()}")
print(f"   Trimestres: {sorted(ipc['TRIMESTRE'].unique())}")
print(f"   Regiones: {ipc['REGION'].nunique()}")

print("\n3. REGISTROS POR REGIÓN:")
print(ipc['REGION'].value_counts().sort_index())

print("\n4. MUESTRA DEL IPC LIMPIO:")
print(ipc.head(20))

# Guardar IPC limpio
ipc.to_csv('ipc_trimestral_limpio.csv', index=False)
print(f"\n✓ IPC limpio guardado en: ipc_trimestral_limpio.csv")

print("\n" + "=" * 80)
print("LIMPIEZA COMPLETADA")
print("=" * 80)