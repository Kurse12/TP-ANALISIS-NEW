import pandas as pd
import os

# Ajusta esta ruta a tu archivo
archivo = 'eph/raw/usu_individual_t117.xlsx'  # CAMBIA ESTO por la ruta real (.xlsx o .xls)

print("=" * 60)
print("DIAGNÓSTICO DE ARCHIVO EPH (EXCEL)")
print("=" * 60)

# Verificar que el archivo existe
if not os.path.exists(archivo):
    print(f"❌ El archivo '{archivo}' no existe")
    print("\nArchivos Excel en el directorio actual:")
    for f in os.listdir('.'):
        if f.endswith(('.xlsx', '.xls', '.XLSX', '.XLS')):
            print(f"  - {f}")
    exit()

print(f"✓ Archivo encontrado: {archivo}")
print(f"✓ Tamaño: {os.path.getsize(archivo) / 1024:.2f} KB\n")

# Cargar el archivo Excel
try:
    # Para .xlsx
    if archivo.endswith('.xlsx'):
        df = pd.read_excel(archivo, engine='openpyxl')
    # Para .xls (formato antiguo)
    else:
        df = pd.read_excel(archivo, engine='xlrd')
    
    print("✓ Archivo Excel cargado exitosamente\n")
    
except Exception as e:
    print(f"❌ Error al cargar: {e}")
    print("\nIntenta instalar las librerías necesarias:")
    print("  pip install openpyxl xlrd")
    exit()

# Mostrar información del archivo cargado
print("=" * 60)
print("INFORMACIÓN DEL ARCHIVO CARGADO")
print("=" * 60)
print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")

# Buscar columna de aglomerado
print("\n" + "=" * 60)
print("CÓDIGOS DE AGLOMERADOS")
print("=" * 60)

aglomerado_encontrado = False
for col in df.columns:
    if 'AGLO' in str(col).upper():
        print(f"\n✓ Columna encontrada: {col}")
        print(f"\nCódigo | Cantidad de registros")
        print("-" * 40)
        conteo = df[col].value_counts().sort_index()
        for codigo, cantidad in conteo.items():
            print(f"{codigo:6} | {cantidad:,}")
        aglomerado_encontrado = True
        break

if not aglomerado_encontrado:
    print("❌ No se encontró columna 'AGLOMERADO'")
    print("\nColumnas disponibles:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2}. {col}")

# Mostrar columnas relacionadas con región o ubicación
print("\n" + "=" * 60)
print("COLUMNAS RELACIONADAS CON UBICACIÓN")
print("=" * 60)
for col in df.columns:
    col_upper = str(col).upper()
    if any(word in col_upper for word in ['AGLO', 'REGION', 'PROV', 'AGLOM']):
        print(f"\n✓ {col}")
        print(f"  Valores únicos: {df[col].nunique()}")
        if df[col].nunique() < 50:
            print(f"  Valores: {sorted(df[col].unique())}")

# Muestra de datos
print("\n" + "=" * 60)
print("MUESTRA DE DATOS (primeras 5 filas)")
print("=" * 60)
print(df.head())

# Información de las columnas principales de EPH
print("\n" + "=" * 60)
print("COLUMNAS PRINCIPALES EPH DETECTADAS")
print("=" * 60)
columnas_clave = ['ANO4', 'TRIMESTRE', 'AGLOMERADO', 'PONDERA', 'ESTADO', 
                  'CH04', 'CH06', 'NIVEL_ED', 'PP04B_COD', 'PP04D_COD', 'P21']
for col in columnas_clave:
    if col in df.columns:
        print(f"✓ {col}")
    else:
        print(f"✗ {col} (no encontrada)")