import pandas as pd


try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    
    print("Buscando archivos shapefile locales...")
    
    # Buscar todos los archivos .shp
    import os
    shapefiles_encontrados = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.shp'):
                path_completo = os.path.join(root, file)
                shapefiles_encontrados.append(path_completo)
    
    if shapefiles_encontrados:
        print(f"\n✓ Shapefiles encontrados: {len(shapefiles_encontrados)}")
        for shp in shapefiles_encontrados:
            print(f"  - {shp}")
        
        # Usar el primero encontrado
        shapefile_path = shapefiles_encontrados[0]
        print(f"\n📂 Cargando: {shapefile_path}")
        
        gdf_completo = gpd.read_file(shapefile_path)
        print(f"✓ Registros cargados: {len(gdf_completo)}")
        print(f"✓ Columnas: {list(gdf_completo.columns)}")
        
        # Identificar columnas relevantes
        print("\n🔍 Buscando CABA y Córdoba...")
        
        # Buscar columnas que contengan nombres
        columnas_texto = [col for col in gdf_completo.columns if gdf_completo[col].dtype == 'object']
        
        # Intentar encontrar CABA y Córdoba
        gdf_filtrado = pd.DataFrame()
        
        for col in columnas_texto:
            # Buscar Ciudad de Buenos Aires / CABA
            mask_caba = gdf_completo[col].astype(str).str.contains(
                'Ciudad.*Buenos Aires|CABA|Ciudad Autónoma', 
                case=False, na=False, regex=True
            )
            
            # Buscar Córdoba + Capital (para filtrar solo la ciudad capital)
            mask_cordoba = gdf_completo[col].astype(str).str.contains(
                'Córdoba', case=False, na=False
            )
            
            if mask_caba.any() or mask_cordoba.any():
                print(f"  ✓ Encontrados en columna '{col}'")
                temp = gdf_completo[mask_caba | mask_cordoba].copy()
                
                # Mostrar qué encontramos
                print(f"    Registros encontrados: {len(temp)}")
                if len(temp) > 0:
                    print(f"    Nombres: {temp[col].unique()}")
                
                if len(gdf_filtrado) == 0:
                    gdf_filtrado = temp
        
        if len(gdf_filtrado) > 0:
            # Crear nombres simplificados
            gdf = gdf_filtrado.copy()
            
            # Asignar nombres simplificados
            gdf['AGLOMERADO_NOMBRE'] = 'Otro'
            
            for col in columnas_texto:
                mask_caba = gdf[col].astype(str).str.contains(
                    'Ciudad.*Buenos Aires|CABA', case=False, na=False, regex=True
                )
                mask_cordoba = gdf[col].astype(str).str.contains('Córdoba', case=False, na=False)
                
                gdf.loc[mask_caba, 'AGLOMERADO_NOMBRE'] = 'CABA'
                gdf.loc[mask_cordoba, 'AGLOMERADO_NOMBRE'] = 'Córdoba'
            
            # Si hay múltiples registros de Córdoba, quedarnos solo con Capital
            if len(gdf[gdf['AGLOMERADO_NOMBRE'] == 'Córdoba']) > 1:
                for col in columnas_texto:
                    if 'capital' in col.lower() or 'depto' in col.lower():
                        mask_capital = gdf[col].astype(str).str.contains('Capital', case=False, na=False)
                        mask_cordoba = gdf['AGLOMERADO_NOMBRE'] == 'Córdoba'
                        gdf = gdf[~mask_cordoba | mask_capital].copy()
                        break
            
            # Disolver geometrías por aglomerado (unir polígonos)
            gdf = gdf.dissolve(by='AGLOMERADO_NOMBRE', as_index=False)
            
            print(f"\n✓ Geometrías finales:")
            print(f"  CABA: {'✓' if 'CABA' in gdf['AGLOMERADO_NOMBRE'].values else '✗'}")
            print(f"  Córdoba: {'✓' if 'Córdoba' in gdf['AGLOMERADO_NOMBRE'].values else '✗'}")
            
            usar_geopandas = True
            
        else:
            print("⚠ No se pudieron identificar CABA y Córdoba automáticamente")
            print("Creando geometrías aproximadas...")
            usar_geopandas = False
            
    else:
        print("⚠ No se encontraron archivos shapefile")
        print("Creando geometrías aproximadas...")
        usar_geopandas = False
    
except ImportError:
    print("⚠ GeoPandas no instalado")
    print("  Para instalar: pip install geopandas")
    usar_geopandas = False
except Exception as e:
    print(f"⚠ Error al cargar shapefile: {str(e)}")
    print("Usando geometrías aproximadas...")
    usar_geopandas = False

# Si no se pudo cargar con geopandas, crear polígonos aproximados
if not usar_geopandas or 'gdf' not in locals():
    print("\n📐 Creando polígonos aproximados...")
    
    try:
        from shapely.geometry import Polygon
        
        # Límites aproximados más realistas
        caba_coords = [
            (-58.531, -34.705), (-58.335, -34.705),
            (-58.335, -34.525), (-58.531, -34.525),
            (-58.531, -34.705)
        ]
        
        cordoba_coords = [
            (-64.350, -31.500), (-64.050, -31.500),
            (-64.050, -31.300), (-64.350, -31.300),
            (-64.350, -31.500)
        ]
        
        gdf = gpd.GeoDataFrame({
            'AGLOMERADO_NOMBRE': ['CABA', 'Córdoba'],
            'geometry': [Polygon(caba_coords), Polygon(cordoba_coords)]
        }, crs='EPSG:4326')
        
        usar_geopandas = True
        print("✓ Polígonos aproximados creados")
        
    except:
        usar_geopandas = False
        import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANÁLISIS AVANZADO - APROBACIÓN DIRECTA")
print("1. Modelo de Imputación de Ingresos")
print("2. Visualizaciones Georreferenciadas")
print("=" * 80)

# ==============================================================================
# 0. CARGA DE DATOS
# ==============================================================================

print("\n" + "=" * 80)
print("CARGANDO DATOS EPH PROCESADOS")
print("=" * 80)

# Opción 1: Cargar desde archivos raw y procesar
from glob import glob

print("\nCargando microdatos EPH desde eph/raw/...")
archivos = glob('eph/raw/*.xlsx') + glob('eph/raw/*.xls')
print(f"Archivos encontrados: {len(archivos)}")

if len(archivos) == 0:
    print("\n⚠ ERROR: No se encontraron archivos en eph/raw/")
    print("Por favor, asegúrate de tener los archivos EPH en la carpeta eph/raw/")
    exit()

# Cargar y consolidar
dfs = []
for i, archivo in enumerate(archivos, 1):
    try:
        if archivo.endswith('.xlsx'):
            df_temp = pd.read_excel(archivo, engine='openpyxl')
        else:
            df_temp = pd.read_excel(archivo, engine='xlrd')
        dfs.append(df_temp)
        if i % 10 == 0:
            print(f"  Cargados: {i}/{len(archivos)}")
    except Exception as e:
        print(f"  ✗ Error en {archivo}: {str(e)[:50]}")

df = pd.concat(dfs, ignore_index=True)
print(f"✓ Total registros consolidados: {len(df):,}")

# Filtrar CABA (33) y Córdoba (7)
df = df[df['AGLOMERADO'].isin([33, 7])].copy()
print(f"✓ Registros CABA + Córdoba: {len(df):,}")

# Crear etiquetas
df['AGLOMERADO_NOMBRE'] = df['AGLOMERADO'].map({33: 'CABA', 7: 'Córdoba'})

# Cargar y aplicar IPC
print("\nCargando IPC...")
ipc = pd.read_csv('ipc_trimestral_limpio.csv')
print(f"✓ IPC cargado: {len(ipc)} registros")

# Verificar que IPC_VAL es numérico
if ipc['IPC_VAL'].dtype == 'object':
    ipc['IPC_VAL'] = ipc['IPC_VAL'].astype(str).str.replace(',', '.').astype(float)

# Mapeo de región
df['REGION_IPC'] = df['REGION'].map({
    1: 'GBA',
    40: 'Noroeste',
    41: 'Noreste', 
    42: 'Cuyo',
    43: 'Pampeana',
    44: 'Patagonia'
})

# Merge con IPC
df = df.merge(ipc[['ANO4', 'TRIMESTRE', 'REGION', 'IPC_VAL']], 
              left_on=['ANO4', 'TRIMESTRE', 'REGION_IPC'], 
              right_on=['ANO4', 'TRIMESTRE', 'REGION'], 
              how='left', suffixes=('', '_ipc'))

# Deflactar ingresos
ipc_base = df['IPC_VAL'].max()
df['P21_REAL'] = np.where(df['IPC_VAL'].notna() & (df['P21'] > 0),
                          (df['P21'] / df['IPC_VAL']) * ipc_base,
                          df['P21'])

print(f"✓ Ingresos deflactados (base IPC = {ipc_base:.2f})")

# ==============================================================================
# PARTE 1: MODELO DE IMPUTACIÓN DE NO RESPUESTA A INGRESOS
# ==============================================================================

print("\n" + "=" * 80)
print("PARTE 1: MODELO DE IMPUTACIÓN DE INGRESOS")
print("=" * 80)

print("\n1.1 PREPARACIÓN DE DATOS PARA MODELADO")
print("-" * 80)

# Filtrar solo ocupados
df_modelo = df[df['ESTADO'] == 1].copy()

# Simular no respuesta (aproximadamente 25% de los casos)
# En datos reales, esto sería donde P21 es NaN
np.random.seed(42)
df_modelo['INGRESO_OBSERVADO'] = df_modelo['P21_REAL'].copy()
mask_missing = np.random.random(len(df_modelo)) < 0.25
df_modelo.loc[mask_missing, 'INGRESO_OBSERVADO'] = np.nan

print(f"Total ocupados: {len(df_modelo):,}")
print(f"Con ingreso declarado: {(~df_modelo['INGRESO_OBSERVADO'].isna()).sum():,}")
print(f"Sin ingreso declarado (no respuesta): {df_modelo['INGRESO_OBSERVADO'].isna().sum():,}")
print(f"Tasa de no respuesta: {(df_modelo['INGRESO_OBSERVADO'].isna().sum() / len(df_modelo) * 100):.2f}%")

# Crear variables para el modelo
print("\n1.2 INGENIERÍA DE CARACTERÍSTICAS")
print("-" * 80)

# Variables dummy
df_modelo['SEXO_VARON'] = (df_modelo['CH04'] == 1).astype(int)
df_modelo['CABA'] = (df_modelo['AGLOMERADO'] == 33).astype(int)

# Edad y edad al cuadrado (capturar no linealidad)
df_modelo['EDAD'] = df_modelo['CH06']
df_modelo['EDAD_CUADRADO'] = df_modelo['CH06'] ** 2

# Nivel educativo (1-7 en EPH)
df_modelo['NIVEL_ED_NUM'] = df_modelo['NIVEL_ED']

# Horas trabajadas (si existe PP07H, sino usar promedio por categoría)
if 'PP07H' in df_modelo.columns:
    df_modelo['HORAS_SEMANALES'] = df_modelo['PP07H']
else:
    df_modelo['HORAS_SEMANALES'] = 40  # Valor por defecto

# Categoría ocupacional
if 'CAT_OCUP' in df_modelo.columns:
    df_modelo['ASALARIADO'] = df_modelo['CAT_OCUP'].isin([3]).astype(int)
else:
    df_modelo['ASALARIADO'] = 1

# Variables temporales
df_modelo['ANIO'] = df_modelo['ANO4']
df_modelo['TRIM'] = df_modelo['TRIMESTRE']

# Transformación logarítmica del ingreso (SOLO PARA VALORES POSITIVOS)
df_modelo = df_modelo[df_modelo['P21_REAL'] > 0].copy()  # Eliminar ingresos <= 0
df_modelo['LOG_INGRESO'] = np.log(df_modelo['P21_REAL'])

# Verificar que no hay infinitos o NaN
print(f"\nVerificación de datos:")
print(f"  Total registros con ingreso positivo: {len(df_modelo):,}")
print(f"  LOG_INGRESO infinitos: {np.isinf(df_modelo['LOG_INGRESO']).sum()}")
print(f"  LOG_INGRESO NaN: {df_modelo['LOG_INGRESO'].isna().sum()}")
print(f"  Rango LOG_INGRESO: [{df_modelo['LOG_INGRESO'].min():.2f}, {df_modelo['LOG_INGRESO'].max():.2f}]")

print("Variables creadas:")
print("  - SEXO_VARON (1=Varón, 0=Mujer)")
print("  - CABA (1=CABA, 0=Córdoba)")
print("  - EDAD y EDAD_CUADRADO")
print("  - NIVEL_ED_NUM (1-7)")
print("  - HORAS_SEMANALES")
print("  - ASALARIADO (1=Sí, 0=No)")
print("  - Variables temporales (ANIO, TRIM)")

# Seleccionar variables para el modelo
variables_x = ['EDAD', 'EDAD_CUADRADO', 'SEXO_VARON', 'NIVEL_ED_NUM', 
               'HORAS_SEMANALES', 'ASALARIADO', 'CABA']

# Preparar datasets (solo con ingresos observados para entrenar)
df_train = df_modelo[~df_modelo['INGRESO_OBSERVADO'].isna()].copy()
df_impute = df_modelo[df_modelo['INGRESO_OBSERVADO'].isna()].copy()

# Eliminar filas con NaN en variables predictoras O en LOG_INGRESO
df_train = df_train.dropna(subset=variables_x + ['LOG_INGRESO'])

# Verificar que no quedan infinitos
df_train = df_train[np.isfinite(df_train['LOG_INGRESO'])].copy()

print(f"\nDatos para entrenamiento (después de limpieza): {len(df_train):,}")
print(f"Datos para imputar: {len(df_impute):,}")

# ==============================================================================
# 1.3 ENTRENAMIENTO DEL MODELO
# ==============================================================================

print("\n1.3 ENTRENAMIENTO Y EVALUACIÓN DEL MODELO")
print("-" * 80)

# Split train/test
X = df_train[variables_x]
y = df_train['LOG_INGRESO']

# Verificación final antes de entrenar
print(f"\nVerificación final de datos de entrenamiento:")
print(f"  Filas en X: {len(X)}")
print(f"  Filas en y: {len(y)}")
print(f"  NaN en X: {X.isna().sum().sum()}")
print(f"  NaN en y: {y.isna().sum()}")
print(f"  Infinitos en y: {np.isinf(y).sum()}")
print(f"  Rango y: [{y.min():.2f}, {y.max():.2f}]")

# Si hay algún problema, eliminarlo
if X.isna().sum().sum() > 0 or y.isna().sum() > 0 or np.isinf(y).sum() > 0:
    print("\n⚠ Limpiando datos problemáticos...")
    mask_validos = ~(X.isna().any(axis=1) | y.isna() | np.isinf(y))
    X = X[mask_validos].copy()
    y = y[mask_validos].copy()
    print(f"  Datos válidos restantes: {len(X):,}")

if len(X) < 100:
    print("\n⚠ ERROR: Muy pocos datos válidos para entrenar el modelo")
    print("Verifica que los archivos EPH tengan datos de ingresos válidos")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# Métricas
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print("MÉTRICAS DE RENDIMIENTO:")
print(f"  R² Entrenamiento: {r2_train:.4f}")
print(f"  R² Test:          {r2_test:.4f}")
print(f"  RMSE Train:       {rmse_train:.4f}")
print(f"  RMSE Test:        {rmse_test:.4f}")
print(f"  MAE Test:         {mae_test:.4f}")

# Coeficientes del modelo
print("\nCOEFICIENTES DEL MODELO (Variable → Log(Ingreso)):")
print("-" * 80)
coef_df = pd.DataFrame({
    'Variable': variables_x,
    'Coeficiente': modelo.coef_,
    'Efecto_%': (np.exp(modelo.coef_) - 1) * 100
})
coef_df = coef_df.sort_values('Coeficiente', ascending=False)
print(coef_df.to_string(index=False))
print(f"\nIntercepto: {modelo.intercept_:.4f}")

print("\nINTERPRETACIÓN DE COEFICIENTES:")
print("-" * 80)
for idx, row in coef_df.iterrows():
    if row['Variable'] == 'SEXO_VARON':
        print(f"  • Ser varón aumenta el ingreso en {row['Efecto_%']:.2f}% (ceteris paribus)")
    elif row['Variable'] == 'CABA':
        print(f"  • Residir en CABA aumenta el ingreso en {row['Efecto_%']:.2f}% vs Córdoba")
    elif row['Variable'] == 'NIVEL_ED_NUM':
        print(f"  • Cada nivel educativo adicional aumenta el ingreso en {row['Efecto_%']:.2f}%")
    elif row['Variable'] == 'EDAD':
        print(f"  • Cada año adicional de edad aumenta el ingreso en {row['Efecto_%']:.2f}%")
    elif row['Variable'] == 'HORAS_SEMANALES':
        print(f"  • Cada hora semanal adicional aumenta el ingreso en {row['Efecto_%']:.2f}%")

# ==============================================================================
# 1.4 IMPUTACIÓN DE VALORES FALTANTES
# ==============================================================================

print("\n1.4 IMPUTACIÓN DE VALORES FALTANTES")
print("-" * 80)

# Predecir ingresos faltantes
if len(df_impute) > 0:
    X_impute = df_impute[variables_x].fillna(df_train[variables_x].mean())
    log_ingreso_imputado = modelo.predict(X_impute)
    ingreso_imputado = np.exp(log_ingreso_imputado)
    
    df_impute['INGRESO_IMPUTADO'] = ingreso_imputado
    df_impute['LOG_INGRESO_IMPUTADO'] = log_ingreso_imputado
    
    print(f"Ingresos imputados: {len(df_impute):,}")
    print(f"Media ingresos imputados: ${df_impute['INGRESO_IMPUTADO'].mean():,.0f}")
    print(f"Media ingresos observados: ${df_train['P21_REAL'].mean():,.0f}")
    print(f"Diferencia: {((df_impute['INGRESO_IMPUTADO'].mean() / df_train['P21_REAL'].mean() - 1) * 100):.2f}%")

# ==============================================================================
# 1.5 VISUALIZACIONES DEL MODELO
# ==============================================================================

print("\n1.5 GENERANDO VISUALIZACIONES DEL MODELO")
print("-" * 80)

fig = plt.figure(figsize=(20, 12))

# 1.5.1 Valores reales vs predichos
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_test, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Log(Ingreso) Real')
ax1.set_ylabel('Log(Ingreso) Predicho')
ax1.set_title(f'Valores Reales vs Predichos\nR² = {r2_test:.4f}')
ax1.grid(True, alpha=0.3)

# 1.5.2 Residuos
ax2 = plt.subplot(2, 3, 2)
residuos = y_test - y_pred_test
ax2.scatter(y_pred_test, residuos, alpha=0.3, s=10)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Log(Ingreso) Predicho')
ax2.set_ylabel('Residuos')
ax2.set_title('Análisis de Residuos')
ax2.grid(True, alpha=0.3)

# 1.5.3 Distribución de residuos
ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuos, bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Residuos')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de Residuos')
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.grid(True, alpha=0.3)

# 1.5.4 Importancia de variables (coeficientes)
ax4 = plt.subplot(2, 3, 4)
coef_plot = coef_df.sort_values('Coeficiente')
colors = ['red' if x < 0 else 'green' for x in coef_plot['Coeficiente']]
ax4.barh(coef_plot['Variable'], coef_plot['Coeficiente'], color=colors, alpha=0.7)
ax4.set_xlabel('Coeficiente')
ax4.set_title('Influencia de Variables en Log(Ingreso)')
ax4.axvline(x=0, color='black', linestyle='-', lw=1)
ax4.grid(True, alpha=0.3, axis='x')

# 1.5.5 Comparación distribuciones: observados vs imputados
ax5 = plt.subplot(2, 3, 5)
if len(df_impute) > 0:
    ax5.hist(df_train['P21_REAL'], bins=50, alpha=0.5, label='Observados', density=True)
    ax5.hist(df_impute['INGRESO_IMPUTADO'], bins=50, alpha=0.5, label='Imputados', density=True)
    ax5.set_xlabel('Ingreso Real ($)')
    ax5.set_ylabel('Densidad')
    ax5.set_title('Distribución: Observados vs Imputados')
    ax5.legend()
    ax5.set_xlim(0, df_train['P21_REAL'].quantile(0.95))
    ax5.grid(True, alpha=0.3)

# 1.5.6 Q-Q plot (normalidad de residuos)
ax6 = plt.subplot(2, 3, 6)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot (Normalidad de Residuos)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('modelo_imputacion_ingresos.png', dpi=300, bbox_inches='tight')
print("✓ Visualizaciones guardadas en 'modelo_imputacion_ingresos.png'")

# ==============================================================================
# PARTE 2: VISUALIZACIONES GEORREFERENCIADAS
# ==============================================================================

print("\n" + "=" * 80)
print("PARTE 2: VISUALIZACIONES GEORREFERENCIADAS")
print("=" * 80)

print("\n2.1 DESCARGA Y CARGA DE GEOMETRÍAS (SHAPEFILE)")
print("-" * 80)

# Intentar cargar shapefiles reales de INDEC
# URL oficial: https://www.indec.gob.ar/indec/web/Nivel4-Tema-1-39-120

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    
    # Intentar cargar shapefile si existe
    print("Buscando archivos shapefile locales...")
    
    # Opción 1: Si tienes el shapefile descargado
    try:
        gdf = gpd.read_file('shapefiles/aglomerados_urbanos.shp')
        print(f"✓ Shapefile cargado: {len(gdf)} aglomerados")
        
        # Filtrar CABA y Córdoba
        gdf = gdf[gdf['AGLOMERADO'].isin([33, 7])].copy()
        gdf['AGLOMERADO_NOMBRE'] = gdf['AGLOMERADO'].map({33: 'CABA', 7: 'Córdoba'})
        
    except:
        print("⚠ Shapefile no encontrado localmente")
        print("Creando geometrías aproximadas basadas en límites conocidos...")
        
        # Opción 2: Crear polígonos aproximados (mientras descargas los reales)
        # Límites aproximados de las ciudades
        caba_coords = [
            (-58.531, -34.705),  # NO
            (-58.335, -34.705),  # NE
            (-58.335, -34.525),  # SE
            (-58.531, -34.525),  # SO
            (-58.531, -34.705)   # Cerrar
        ]
        
        cordoba_coords = [
            (-64.350, -31.500),  # NO
            (-64.050, -31.500),  # NE
            (-64.050, -31.300),  # SE
            (-64.350, -31.300),  # SO
            (-64.350, -31.500)   # Cerrar
        ]
        
        gdf = gpd.GeoDataFrame({
            'AGLOMERADO': [33, 7],
            'AGLOMERADO_NOMBRE': ['CABA', 'Córdoba'],
            'geometry': [Polygon(caba_coords), Polygon(cordoba_coords)]
        }, crs='EPSG:4326')
        
        print("✓ Geometrías aproximadas creadas")
        print("📥 Para mapas reales, descarga shapefiles de:")
        print("   https://www.indec.gob.ar/ftp/cuadros/territorio/codgeo/Codgeo_Pais_x_dpto_con_datos.zip")
    
    usar_geopandas = True
    
except ImportError:
    print("⚠ GeoPandas no instalado. Usando mapas simplificados.")
    print("   Para instalar: pip install geopandas")
    usar_geopandas = False

# Calcular indicadores por jurisdicción para el último trimestre
ultimo_trim = df['ANO4'].max()
ultimo_t = df[df['ANO4'] == ultimo_trim]['TRIMESTRE'].max()

datos_mapa = df[(df['ANO4'] == ultimo_trim) & (df['TRIMESTRE'] == ultimo_t)].groupby('AGLOMERADO_NOMBRE').apply(
    lambda x: pd.Series({
        'desocupacion': (x[x['ESTADO'] == 2]['PONDERA'].sum() / 
                        x[x['ESTADO'].isin([1,2])]['PONDERA'].sum() * 100),
        'empleo': (x[x['ESTADO'] == 1]['PONDERA'].sum() / x['PONDERA'].sum() * 100),
        'ingreso_medio': x[x['ESTADO'] == 1]['P21_REAL'].mean(),
        'pob_total': x['PONDERA'].sum()
    })
).reset_index()

print(f"\nDatos para mapeo (período {ultimo_trim}-T{ultimo_t}):")
print(datos_mapa)

# ==============================================================================
# 2.2 MAPAS TEMÁTICOS CON GEOMETRÍAS REALES
# ==============================================================================

print("\n2.2 GENERANDO MAPAS TEMÁTICOS CON GEOMETRÍAS")
print("-" * 80)

if usar_geopandas:
    # MAPAS CON GEOPANDAS (GEOMETRÍAS REALES)
    
    # Merge datos con geometrías
    gdf_map = gdf.merge(datos_mapa, on='AGLOMERADO_NOMBRE', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 2.2.1 Mapa de Tasa de Desocupación
    ax1 = axes[0, 0]
    gdf_map.plot(column='desocupacion', ax=ax1, legend=True,
                 cmap='Reds', edgecolor='black', linewidth=2,
                 legend_kwds={'label': 'Tasa de Desocupación (%)', 'shrink': 0.6})
    
    # Añadir etiquetas
    for idx, row in gdf_map.iterrows():
        centroid = row.geometry.centroid
        ax1.annotate(f"{row['AGLOMERADO_NOMBRE']}\n{row['desocupacion']:.2f}%",
                    xy=(centroid.x, centroid.y), ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title(f'Tasa de Desocupación (%)\nPeríodo: {ultimo_trim}-T{ultimo_t}',
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax1.grid(True, alpha=0.3)
    
    # 2.2.2 Mapa de Tasa de Empleo
    ax2 = axes[0, 1]
    gdf_map.plot(column='empleo', ax=ax2, legend=True,
                 cmap='Greens', edgecolor='black', linewidth=2,
                 legend_kwds={'label': 'Tasa de Empleo (%)', 'shrink': 0.6})
    
    for idx, row in gdf_map.iterrows():
        centroid = row.geometry.centroid
        ax2.annotate(f"{row['AGLOMERADO_NOMBRE']}\n{row['empleo']:.2f}%",
                    xy=(centroid.x, centroid.y), ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title(f'Tasa de Empleo (%)\nPeríodo: {ultimo_trim}-T{ultimo_t}',
                  fontsize=16, fontweight='bold')
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.grid(True, alpha=0.3)
    
    # 2.2.3 Mapa de Ingreso Medio
    ax3 = axes[1, 0]
    gdf_map.plot(column='ingreso_medio', ax=ax3, legend=True,
                 cmap='Blues', edgecolor='black', linewidth=2,
                 legend_kwds={'label': 'Ingreso Real ($)', 'shrink': 0.6})
    
    for idx, row in gdf_map.iterrows():
        centroid = row.geometry.centroid
        ax3.annotate(f"{row['AGLOMERADO_NOMBRE']}\n${row['ingreso_medio']:,.0f}",
                    xy=(centroid.x, centroid.y), ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_title(f'Ingreso Real Medio ($)\nPeríodo: {ultimo_trim}-T{ultimo_t}',
                  fontsize=16, fontweight='bold')
    ax3.set_xlabel('Longitud')
    ax3.set_ylabel('Latitud')
    ax3.grid(True, alpha=0.3)
    
    # 2.2.4 Mapa base con contexto (Argentina)
    ax4 = axes[1, 1]
    
    # Cargar mapa base de Argentina si está disponible
    try:
        argentina = gpd.read_file('shapefiles/provincias.shp')
        argentina.plot(ax=ax4, color='lightgray', edgecolor='black', alpha=0.3)
    except:
        pass
    
    # Plotear ciudades con color por desocupación y tamaño por población
    gdf_map.plot(column='desocupacion', ax=ax4, cmap='RdYlGn_r',
                 edgecolor='black', linewidth=2, alpha=0.7)
    
    for idx, row in gdf_map.iterrows():
        centroid = row.geometry.centroid
        ax4.annotate(f"{row['AGLOMERADO_NOMBRE']}\nDesoc: {row['desocupacion']:.1f}%\nIng: ${row['ingreso_medio']/1000:.0f}k",
                    xy=(centroid.x, centroid.y), ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax4.set_title(f'Mapa Integrado: Desocupación e Ingresos\nPeríodo: {ultimo_trim}-T{ultimo_t}',
                  fontsize=16, fontweight='bold')
    ax4.set_xlabel('Longitud')
    ax4.set_ylabel('Latitud')
    ax4.set_xlim(-70, -53)
    ax4.set_ylim(-40, -25)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mapas_georreferenciados_reales.png', dpi=300, bbox_inches='tight')
    print("✓ Mapas con geometrías reales guardados en 'mapas_georreferenciados_reales.png'")

else:
    # FALLBACK: Mapas simplificados (código anterior)
    print("⚠ Usando visualización simplificada sin geometrías")
    
    # Coordenadas para mapas simplificados
    ciudades_coords = {
        'CABA': {'lat': -34.6037, 'lon': -58.3816, 'bbox': [-58.53, -58.33, -34.70, -34.52]},
        'Córdoba': {'lat': -31.4201, 'lon': -64.1888, 'bbox': [-64.35, -64.05, -31.50, -31.30]}
    }
    
    fig = plt.figure(figsize=(20, 15))
    
    # Función auxiliar para dibujar ciudades
    def dibujar_ciudad(ax, ciudad, valor, cmap, vmin, vmax, titulo):
        coords = ciudades_coords[ciudad]
        bbox = coords['bbox']
        
        # Dibujar rectángulo de la ciudad
        color_norm = (valor - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        color = plt.cm.get_cmap(cmap)(color_norm)
        
        rect = Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2],
                         facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Etiqueta
        ax.text(coords['lon'], coords['lat'], ciudad, 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Valor
        ax.text(coords['lon'], coords['lat']-0.15, f'{valor:.2f}',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 2.2.1 Mapa de Tasa de Desocupación
    ax1 = plt.subplot(2, 2, 1)
    vmin_d = datos_mapa['desocupacion'].min()
    vmax_d = datos_mapa['desocupacion'].max()
    for _, row in datos_mapa.iterrows():
        dibujar_ciudad(ax1, row['AGLOMERADO_NOMBRE'], row['desocupacion'], 'Reds', vmin_d, vmax_d, 'Desocupación')

    ax1.set_xlim(-66, -57)
    ax1.set_ylim(-35, -30)
    ax1.set_xlabel('Longitud')
    ax1.set_ylabel('Latitud')
    ax1.set_title('Tasa de Desocupación (%)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Colorbar
    sm1 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin_d, vmax=vmax_d))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label('Tasa de Desocupación (%)', rotation=270, labelpad=20)

    # 2.2.2 Mapa de Tasa de Empleo
    ax2 = plt.subplot(2, 2, 2)
    vmin_e = datos_mapa['empleo'].min()
    vmax_e = datos_mapa['empleo'].max()
    for _, row in datos_mapa.iterrows():
        dibujar_ciudad(ax2, row['AGLOMERADO_NOMBRE'], row['empleo'], 'Greens', vmin_e, vmax_e, 'Empleo')

    ax2.set_xlim(-66, -57)
    ax2.set_ylim(-35, -30)
    ax2.set_xlabel('Longitud')
    ax2.set_ylabel('Latitud')
    ax2.set_title('Tasa de Empleo (%)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
                  fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    sm2 = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin_e, vmax=vmax_e))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', pad=0.02)
    cbar2.set_label('Tasa de Empleo (%)', rotation=270, labelpad=20)

    # 2.2.3 Mapa de Ingreso Medio
    ax3 = plt.subplot(2, 2, 3)
    vmin_i = datos_mapa['ingreso_medio'].min()
    vmax_i = datos_mapa['ingreso_medio'].max()
    for _, row in datos_mapa.iterrows():
        dibujar_ciudad(ax3, row['AGLOMERADO_NOMBRE'], row['ingreso_medio'], 'Blues', vmin_i, vmax_i, 'Ingreso')

    ax3.set_xlim(-66, -57)
    ax3.set_ylim(-35, -30)
    ax3.set_xlabel('Longitud')
    ax3.set_ylabel('Latitud')
    ax3.set_title('Ingreso Real Medio ($)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
                  fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    sm3 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin_i, vmax=vmax_i))
    sm3.set_array([])
    cbar3 = plt.colorbar(sm3, ax=ax3, orientation='vertical', pad=0.02)
    cbar3.set_label('Ingreso Real ($)', rotation=270, labelpad=20)

    # 2.2.4 Mapa combinado (Desocupación vs Ingreso)
    ax4 = plt.subplot(2, 2, 4)
    for _, row in datos_mapa.iterrows():
        coords = ciudades_coords[row['AGLOMERADO_NOMBRE']]
        
        # Tamaño proporcional a población
        size = (row['pob_total'] / datos_mapa['pob_total'].max()) * 5000
        
        # Color por desocupación
        color_norm = (row['desocupacion'] - vmin_d) / (vmax_d - vmin_d) if vmax_d > vmin_d else 0.5
        color = plt.cm.get_cmap('RdYlGn_r')(color_norm)
        
        ax4.scatter(coords['lon'], coords['lat'], s=size, c=[color], 
                   alpha=0.6, edgecolors='black', linewidth=2)
        
        ax4.text(coords['lon'], coords['lat']-0.3, row['AGLOMERADO_NOMBRE'],
                ha='center', va='top', fontsize=12, fontweight='bold')
        
        ax4.text(coords['lon'], coords['lat']+0.3, 
                f"Desoc: {row['desocupacion']:.1f}%\nIng: ${row['ingreso_medio']/1000:.0f}k",
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax4.set_xlim(-66, -57)
    ax4.set_ylim(-35, -30)
    ax4.set_xlabel('Longitud')
    ax4.set_ylabel('Latitud')
    ax4.set_title('Mapa Integrado: Desocupación e Ingresos\n(Tamaño = Población)', 
                  fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(['Tamaño: Población', 'Color: Desocupación (rojo=alto)'], loc='upper left')

    plt.tight_layout()
    plt.savefig('mapas_georreferenciados.png', dpi=300, bbox_inches='tight')
    print("✓ Mapas guardados en 'mapas_georreferenciados.png'")

fig = plt.figure(figsize=(20, 15))

# Función auxiliar para dibujar ciudades
def dibujar_ciudad(ax, ciudad, valor, cmap, vmin, vmax, titulo):
    coords = ciudades_coords[ciudad]
    bbox = coords['bbox']
    
    # Dibujar rectángulo de la ciudad
    color_norm = (valor - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    color = plt.cm.get_cmap(cmap)(color_norm)
    
    rect = Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2],
                     facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
    ax.add_patch(rect)
    
    # Etiqueta
    ax.text(coords['lon'], coords['lat'], ciudad, 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Valor
    ax.text(coords['lon'], coords['lat']-0.15, f'{valor:.2f}',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2.2.1 Mapa de Tasa de Desocupación
ax1 = plt.subplot(2, 2, 1)
vmin_d = datos_mapa['desocupacion'].min()
vmax_d = datos_mapa['desocupacion'].max()
for _, row in datos_mapa.iterrows():
    dibujar_ciudad(ax1, row['AGLOMERADO_NOMBRE'], row['desocupacion'], 'Reds', vmin_d, vmax_d, 'Desocupación')

ax1.set_xlim(-66, -57)
ax1.set_ylim(-35, -30)
ax1.set_xlabel('Longitud')
ax1.set_ylabel('Latitud')
ax1.set_title('Tasa de Desocupación (%)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Colorbar
sm1 = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=vmin_d, vmax=vmax_d))
sm1.set_array([])
cbar1 = plt.colorbar(sm1, ax=ax1, orientation='vertical', pad=0.02)
cbar1.set_label('Tasa de Desocupación (%)', rotation=270, labelpad=20)

# 2.2.2 Mapa de Tasa de Empleo
ax2 = plt.subplot(2, 2, 2)
vmin_e = datos_mapa['empleo'].min()
vmax_e = datos_mapa['empleo'].max()
for _, row in datos_mapa.iterrows():
    dibujar_ciudad(ax2, row['AGLOMERADO_NOMBRE'], row['empleo'], 'Greens', vmin_e, vmax_e, 'Empleo')

ax2.set_xlim(-66, -57)
ax2.set_ylim(-35, -30)
ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
ax2.set_title('Tasa de Empleo (%)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
              fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.3)

sm2 = plt.cm.ScalarMappable(cmap='Greens', norm=plt.Normalize(vmin=vmin_e, vmax=vmax_e))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax2, orientation='vertical', pad=0.02)
cbar2.set_label('Tasa de Empleo (%)', rotation=270, labelpad=20)

# 2.2.3 Mapa de Ingreso Medio
ax3 = plt.subplot(2, 2, 3)
vmin_i = datos_mapa['ingreso_medio'].min()
vmax_i = datos_mapa['ingreso_medio'].max()
for _, row in datos_mapa.iterrows():
    dibujar_ciudad(ax3, row['AGLOMERADO_NOMBRE'], row['ingreso_medio'], 'Blues', vmin_i, vmax_i, 'Ingreso')

ax3.set_xlim(-66, -57)
ax3.set_ylim(-35, -30)
ax3.set_xlabel('Longitud')
ax3.set_ylabel('Latitud')
ax3.set_title('Ingreso Real Medio ($)\n' + f'Período: {ultimo_trim}-T{ultimo_t}', 
              fontsize=16, fontweight='bold')
ax3.grid(True, alpha=0.3)

sm3 = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin_i, vmax=vmax_i))
sm3.set_array([])
cbar3 = plt.colorbar(sm3, ax=ax3, orientation='vertical', pad=0.02)
cbar3.set_label('Ingreso Real ($)', rotation=270, labelpad=20)

# 2.2.4 Mapa combinado (Desocupación vs Ingreso)
ax4 = plt.subplot(2, 2, 4)
for _, row in datos_mapa.iterrows():
    coords = ciudades_coords[row['AGLOMERADO_NOMBRE']]
    
    # Tamaño proporcional a población
    size = (row['pob_total'] / datos_mapa['pob_total'].max()) * 5000
    
    # Color por desocupación
    color_norm = (row['desocupacion'] - vmin_d) / (vmax_d - vmin_d) if vmax_d > vmin_d else 0.5
    color = plt.cm.get_cmap('RdYlGn_r')(color_norm)
    
    ax4.scatter(coords['lon'], coords['lat'], s=size, c=[color], 
               alpha=0.6, edgecolors='black', linewidth=2)
    
    ax4.text(coords['lon'], coords['lat']-0.3, row['AGLOMERADO_NOMBRE'],
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    ax4.text(coords['lon'], coords['lat']+0.3, 
            f"Desoc: {row['desocupacion']:.1f}%\nIng: ${row['ingreso_medio']/1000:.0f}k",
            ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax4.set_xlim(-66, -57)
ax4.set_ylim(-35, -30)
ax4.set_xlabel('Longitud')
ax4.set_ylabel('Latitud')
ax4.set_title('Mapa Integrado: Desocupación e Ingresos\n(Tamaño = Población)', 
              fontsize=16, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(['Tamaño: Población', 'Color: Desocupación (rojo=alto)'], loc='upper left')

plt.tight_layout()
plt.savefig('mapas_georreferenciados.png', dpi=300, bbox_inches='tight')
print("✓ Mapas guardados en 'mapas_georreferenciados.png'")

# ==============================================================================
# 2.3 MAPA TEMPORAL (EVOLUCIÓN EN EL TIEMPO)
# ==============================================================================

print("\n2.3 GENERANDO MAPA DE EVOLUCIÓN TEMPORAL")
print("-" * 80)

# Seleccionar 4 períodos clave
periodos_clave = [
    (2016, 2), (2020, 3), (2023, 3), (ultimo_trim, ultimo_t)
]

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, (anio, trim) in enumerate(periodos_clave):
    ax = axes[idx]
    
    datos_periodo = df[(df['ANO4'] == anio) & (df['TRIMESTRE'] == trim)].groupby('AGLOMERADO_NOMBRE').apply(
        lambda x: pd.Series({
            'desocupacion': (x[x['ESTADO'] == 2]['PONDERA'].sum() / 
                            x[x['ESTADO'].isin([1,2])]['PONDERA'].sum() * 100),
            'ingreso_medio': x[x['ESTADO'] == 1]['P21_REAL'].mean()
        })
    ).reset_index()
    
    for _, row in datos_periodo.iterrows():
        coords = ciudades_coords[row['AGLOMERADO_NOMBRE']]
        bbox = coords['bbox']
        
        # Color por desocupación
        color_norm = row['desocupacion'] / 15  # Normalizar a escala 0-15%
        color = plt.cm.get_cmap('RdYlGn_r')(min(color_norm, 1))
        
        rect = Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2],
                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        ax.text(coords['lon'], coords['lat'], row['AGLOMERADO_NOMBRE'],
               ha='center', va='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.text(coords['lon'], coords['lat']-0.2, f"{row['desocupacion']:.1f}%",
               ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlim(-66, -57)
    ax.set_ylim(-35, -30)
    ax.set_xlabel('Longitud', fontsize=10)
    ax.set_ylabel('Latitud', fontsize=10)
    ax.set_title(f'Tasa de Desocupación - {anio} T{trim}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.suptitle('Evolución Temporal de la Desocupación (Georreferenciada)', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('evolucion_temporal_georreferenciada.png', dpi=300, bbox_inches='tight')
print("✓ Mapa temporal guardado en 'evolucion_temporal_georreferenciada.png'")

# ==============================================================================
# EXPORTAR RESULTADOS DEL MODELO
# ==============================================================================

print("\n2.4 EXPORTANDO RESULTADOS")
print("-" * 80)

# Crear tabla de resultados del modelo
resultados_modelo = pd.DataFrame({
    'Métrica': ['R² Train', 'R² Test', 'RMSE Train', 'RMSE Test', 'MAE Test'],
    'Valor': [r2_train, r2_test, rmse_train, rmse_test, mae_test]
})

# Exportar a Excel
with pd.ExcelWriter('resultados_aprobacion_directa.xlsx', engine='openpyxl') as writer:
    resultados_modelo.to_excel(writer, sheet_name='Metricas_Modelo', index=False)
    coef_df.to_excel(writer, sheet_name='Coeficientes', index=False)
    datos_mapa.to_excel(writer, sheet_name='Datos_Mapas', index=False)
    if len(df_impute) > 0:
        df_impute[['CH06', 'CH04', 'NIVEL_ED', 'P21_REAL', 'INGRESO_IMPUTADO']].head(100).to_excel(
            writer, sheet_name='Ejemplo_Imputaciones', index=False
        )

print("✓ Resultados exportados a 'resultados_aprobacion_directa.xlsx'")

print("\n" + "=" * 80)
print("✓ ANÁLISIS DE APROBACIÓN DIRECTA COMPLETADO")
print("=" * 80)
print("\nArchivos generados:")
print("  1. modelo_imputacion_ingresos.png")
print("  2. mapas_georreferenciados.png")
print("  3. evolucion_temporal_georreferenciada.png")
print("  4. resultados_aprobacion_directa.xlsx")
print("=" * 80)