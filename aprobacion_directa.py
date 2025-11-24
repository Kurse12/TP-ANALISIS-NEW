import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matplotlib.patches import Rectangle
from glob import glob
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANÁLISIS AVANZADO - APROBACIÓN DIRECTA")
print("1. Modelo de Imputación de Ingresos")
print("2. Visualizaciones Georreferenciadas")
print("=" * 80)

# ==============================================================================
# 0. CARGA DE DATOS EPH
# ==============================================================================

print("\n" + "=" * 80)
print("CARGANDO DATOS EPH PROCESADOS")
print("=" * 80)

print("\nCargando microdatos EPH desde eph/raw/...")
archivos = glob('eph/raw/*.xlsx') + glob('eph/raw/*.xls')
print(f"Archivos encontrados: {len(archivos)}")

if len(archivos) == 0:
    print("\n⚠ ERROR: No se encontraron archivos en eph/raw/")
    exit()

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
print(f"✓ Total registros: {len(df):,}")

df = df[df['AGLOMERADO'].isin([33, 7])].copy()
print(f"✓ CABA + Córdoba: {len(df):,}")

df['AGLOMERADO_NOMBRE'] = df['AGLOMERADO'].map({33: 'CABA', 7: 'Córdoba'})

# Cargar IPC
print("\nCargando IPC...")
ipc = pd.read_csv('ipc_trimestral_limpio.csv')
if ipc['IPC_VAL'].dtype == 'object':
    ipc['IPC_VAL'] = ipc['IPC_VAL'].astype(str).str.replace(',', '.').astype(float)

df['REGION_IPC'] = df['REGION'].map({
    1: 'GBA', 40: 'Noroeste', 41: 'Noreste', 42: 'Cuyo', 43: 'Pampeana', 44: 'Patagonia'
})

df = df.merge(ipc[['ANO4', 'TRIMESTRE', 'REGION', 'IPC_VAL']], 
              left_on=['ANO4', 'TRIMESTRE', 'REGION_IPC'], 
              right_on=['ANO4', 'TRIMESTRE', 'REGION'], 
              how='left', suffixes=('', '_ipc'))

ipc_base = df['IPC_VAL'].max()
df['P21_REAL'] = np.where(df['IPC_VAL'].notna() & (df['P21'] > 0),
                          (df['P21'] / df['IPC_VAL']) * ipc_base, df['P21'])

print(f"✓ Ingresos deflactados")

# ==============================================================================
# PARTE 1: MODELO DE IMPUTACIÓN
# ==============================================================================

print("\n" + "=" * 80)
print("PARTE 1: MODELO DE IMPUTACIÓN DE INGRESOS")
print("=" * 80)

df_modelo = df[df['ESTADO'] == 1].copy()

np.random.seed(42)
df_modelo['INGRESO_OBSERVADO'] = df_modelo['P21_REAL'].copy()
mask_missing = np.random.random(len(df_modelo)) < 0.25
df_modelo.loc[mask_missing, 'INGRESO_OBSERVADO'] = np.nan

print(f"\nTotal ocupados: {len(df_modelo):,}")
print(f"Tasa de no respuesta: {(mask_missing.sum() / len(df_modelo) * 100):.2f}%")

# Variables
df_modelo['SEXO_VARON'] = (df_modelo['CH04'] == 1).astype(int)
df_modelo['CABA'] = (df_modelo['AGLOMERADO'] == 33).astype(int)
df_modelo['EDAD'] = df_modelo['CH06']
df_modelo['EDAD_CUADRADO'] = df_modelo['CH06'] ** 2
df_modelo['NIVEL_ED_NUM'] = df_modelo['NIVEL_ED']
df_modelo['HORAS_SEMANALES'] = df_modelo.get('PP07H', 40)
df_modelo['ASALARIADO'] = df_modelo.get('CAT_OCUP', 3).isin([3]).astype(int)

df_modelo = df_modelo[df_modelo['P21_REAL'] > 0].copy()
df_modelo['LOG_INGRESO'] = np.log(df_modelo['P21_REAL'])

variables_x = ['EDAD', 'EDAD_CUADRADO', 'SEXO_VARON', 'NIVEL_ED_NUM', 
               'HORAS_SEMANALES', 'ASALARIADO', 'CABA']

df_train = df_modelo[~df_modelo['INGRESO_OBSERVADO'].isna()].copy()
df_impute = df_modelo[df_modelo['INGRESO_OBSERVADO'].isna()].copy()

df_train = df_train.dropna(subset=variables_x + ['LOG_INGRESO'])
df_train = df_train[np.isfinite(df_train['LOG_INGRESO'])].copy()

print(f"Datos entrenamiento: {len(df_train):,}")

# Entrenar
X = df_train[variables_x]
y = df_train['LOG_INGRESO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred_test = modelo.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\nMÉTRICAS:")
print(f"  R²: {r2_test:.4f}")
print(f"  RMSE: {rmse_test:.4f}")
print(f"  MAE: {mae_test:.4f}")

coef_df = pd.DataFrame({
    'Variable': variables_x,
    'Coeficiente': modelo.coef_,
    'Efecto_%': (np.exp(modelo.coef_) - 1) * 100
})

# Imputar
if len(df_impute) > 0:
    X_impute = df_impute[variables_x].fillna(df_train[variables_x].mean())
    df_impute['INGRESO_IMPUTADO'] = np.exp(modelo.predict(X_impute))
    print(f"\nIngresos imputados: {len(df_impute):,}")

# Gráficos modelo
fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_pred_test, alpha=0.3, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_title(f'Reales vs Predichos\nR² = {r2_test:.4f}')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
residuos = y_test - y_pred_test
ax2.scatter(y_pred_test, residuos, alpha=0.3, s=10)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_title('Residuos')
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.hist(residuos, bins=50, edgecolor='black', alpha=0.7)
ax3.set_title('Distribución Residuos')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
coef_plot = coef_df.sort_values('Coeficiente')
colors = ['red' if x < 0 else 'green' for x in coef_plot['Coeficiente']]
ax4.barh(coef_plot['Variable'], coef_plot['Coeficiente'], color=colors, alpha=0.7)
ax4.set_title('Influencia Variables')
ax4.axvline(x=0, color='black')
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
if len(df_impute) > 0:
    ax5.hist(df_train['P21_REAL'], bins=50, alpha=0.5, label='Observados', density=True)
    ax5.hist(df_impute['INGRESO_IMPUTADO'], bins=50, alpha=0.5, label='Imputados', density=True)
    ax5.legend()
    ax5.set_title('Observados vs Imputados')
    ax5.set_xlim(0, df_train['P21_REAL'].quantile(0.95))

ax6 = plt.subplot(2, 3, 6)
from scipy import stats
stats.probplot(residuos, dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot')

plt.tight_layout()
plt.savefig('modelo_imputacion_ingresos.png', dpi=300, bbox_inches='tight')
print("✓ Gráficos modelo guardados")

# ==============================================================================
# PARTE 2: MAPAS GEORREFERENCIADOS
# ==============================================================================

print("\n" + "=" * 80)
print("PARTE 2: VISUALIZACIONES GEORREFERENCIADAS")
print("=" * 80)

# Datos para mapas
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

print(f"\nDatos período {ultimo_trim}-T{ultimo_t}:")
print(datos_mapa)

# Intentar cargar shapefiles
usar_geopandas = False
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    import os
    
    print("\nBuscando shapefiles...")
    shps = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.shp'):
                shps.append(os.path.join(root, file))
    
    if shps:
        print(f"✓ Encontrados: {len(shps)} archivos")
        gdf_completo = gpd.read_file(shps[0])
        print(f"✓ Cargado: {len(gdf_completo)} registros")
        print(f"✓ Columnas: {list(gdf_completo.columns)[:5]}...")
        
        # Buscar CABA y Córdoba
        for col in gdf_completo.columns:
            if gdf_completo[col].dtype == 'object':
                mask_caba = gdf_completo[col].astype(str).str.contains(
                    'Ciudad.*Buenos Aires|CABA', case=False, na=False, regex=True)
                mask_cordoba = gdf_completo[col].astype(str).str.contains('Córdoba', case=False, na=False)
                
                if mask_caba.any() or mask_cordoba.any():
                    print(f"✓ Encontrados en '{col}'")
                    gdf = gdf_completo[mask_caba | mask_cordoba].copy()
                    gdf['AGLOMERADO_NOMBRE'] = 'Otro'
                    gdf.loc[mask_caba, 'AGLOMERADO_NOMBRE'] = 'CABA'
                    gdf.loc[mask_cordoba, 'AGLOMERADO_NOMBRE'] = 'Córdoba'
                    
                    if len(gdf[gdf['AGLOMERADO_NOMBRE'] == 'Córdoba']) > 1:
                        for c in gdf.columns:
                            if gdf[c].dtype == 'object':
                                mask_cap = gdf[c].astype(str).str.contains('Capital', case=False, na=False)
                                if mask_cap.any():
                                    gdf = gdf[~(gdf['AGLOMERADO_NOMBRE'] == 'Córdoba') | mask_cap].copy()
                                    break
                    
                    gdf = gdf.dissolve(by='AGLOMERADO_NOMBRE', as_index=False)
                    usar_geopandas = True
                    print(f"✓ Geometrías: CABA={'✓' if 'CABA' in gdf['AGLOMERADO_NOMBRE'].values else '✗'}, Córdoba={'✓' if 'Córdoba' in gdf['AGLOMERADO_NOMBRE'].values else '✗'}")
                    break
    
    if not usar_geopandas:
        print("⚠ No se identificaron automáticamente, creando polígonos...")
        caba_coords = [(-58.531, -34.705), (-58.335, -34.705), (-58.335, -34.525), (-58.531, -34.525), (-58.531, -34.705)]
        cordoba_coords = [(-64.350, -31.500), (-64.050, -31.500), (-64.050, -31.300), (-64.350, -31.300), (-64.350, -31.500)]
        
        gdf = gpd.GeoDataFrame({
            'AGLOMERADO_NOMBRE': ['CABA', 'Córdoba'],
            'geometry': [Polygon(caba_coords), Polygon(cordoba_coords)]
        }, crs='EPSG:4326')
        usar_geopandas = True

except Exception as e:
    print(f"⚠ Error: {str(e)[:100]}")
    usar_geopandas = False

# Generar mapas
if usar_geopandas and 'gdf' in locals():
    gdf_map = gdf.merge(datos_mapa, on='AGLOMERADO_NOMBRE', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    gdf_map.plot(column='desocupacion', ax=axes[0,0], legend=True, cmap='Reds', edgecolor='black', linewidth=2)
    axes[0,0].set_title(f'Desocupación (%)\n{ultimo_trim}-T{ultimo_t}', fontweight='bold')
    
    gdf_map.plot(column='empleo', ax=axes[0,1], legend=True, cmap='Greens', edgecolor='black', linewidth=2)
    axes[0,1].set_title(f'Empleo (%)\n{ultimo_trim}-T{ultimo_t}', fontweight='bold')
    
    gdf_map.plot(column='ingreso_medio', ax=axes[1,0], legend=True, cmap='Blues', edgecolor='black', linewidth=2)
    axes[1,0].set_title(f'Ingreso Real ($)\n{ultimo_trim}-T{ultimo_t}', fontweight='bold')
    
    gdf_map.plot(column='desocupacion', ax=axes[1,1], cmap='RdYlGn_r', edgecolor='black', linewidth=2)
    axes[1,1].set_title(f'Integrado\n{ultimo_trim}-T{ultimo_t}', fontweight='bold')
    
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mapas_georreferenciados_reales.png', dpi=300, bbox_inches='tight')
    print("✓ Mapas con geometrías reales guardados")

else:
    # Mapas simplificados
    ciudades_coords = {
        'CABA': {'lat': -34.6037, 'lon': -58.3816, 'bbox': [-58.53, -58.33, -34.70, -34.52]},
        'Córdoba': {'lat': -31.4201, 'lon': -64.1888, 'bbox': [-64.35, -64.05, -31.50, -31.30]}
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for idx, (var, cmap, label) in enumerate([
        ('desocupacion', 'Reds', 'Desocupación (%)'),
        ('empleo', 'Greens', 'Empleo (%)'),
        ('ingreso_medio', 'Blues', 'Ingreso ($)'),
        (None, None, 'Integrado')
    ]):
        ax = axes[idx]
        
        if var:
            vmin, vmax = datos_mapa[var].min(), datos_mapa[var].max()
            for _, row in datos_mapa.iterrows():
                coords = ciudades_coords[row['AGLOMERADO_NOMBRE']]
                bbox = coords['bbox']
                color_norm = (row[var] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                color = plt.cm.get_cmap(cmap)(color_norm)
                rect = Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2],
                                facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
                ax.add_patch(rect)
                ax.text(coords['lon'], coords['lat'], row['AGLOMERADO_NOMBRE'],
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(coords['lon'], coords['lat']-0.15, f'{row[var]:.1f}',
                       ha='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=label)
        else:
            for _, row in datos_mapa.iterrows():
                coords = ciudades_coords[row['AGLOMERADO_NOMBRE']]
                size = (row['pob_total'] / datos_mapa['pob_total'].max()) * 5000
                ax.scatter(coords['lon'], coords['lat'], s=size, alpha=0.6, edgecolors='black', linewidth=2)
                ax.text(coords['lon'], coords['lat']-0.3, row['AGLOMERADO_NOMBRE'], ha='center', fontweight='bold')
        
        ax.set_xlim(-66, -57)
        ax.set_ylim(-35, -30)
        ax.set_title(f'{label}\n{ultimo_trim}-T{ultimo_t}' if var else f'Integrado\n{ultimo_trim}-T{ultimo_t}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mapas_georreferenciados.png', dpi=300, bbox_inches='tight')
    print("✓ Mapas simplificados guardados")

# Exportar
resultados_modelo = pd.DataFrame({
    'Métrica': ['R² Test', 'RMSE Test', 'MAE Test'],
    'Valor': [r2_test, rmse_test, mae_test]
})

with pd.ExcelWriter('resultados_aprobacion_directa.xlsx', engine='openpyxl') as writer:
    resultados_modelo.to_excel(writer, sheet_name='Metricas', index=False)
    coef_df.to_excel(writer, sheet_name='Coeficientes', index=False)
    datos_mapa.to_excel(writer, sheet_name='Datos_Mapas', index=False)

print("\n" + "=" * 80)
print("✓ ANÁLISIS COMPLETADO")
print("=" * 80)
print("\nArchivos generados:")
print("  1. modelo_imputacion_ingresos.png")
print("  2. mapas_georreferenciados.png (o mapas_georreferenciados_reales.png)")
print("  3. resultados_aprobacion_directa.xlsx")
print("=" * 80)