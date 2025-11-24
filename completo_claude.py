import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("ANÁLISIS EPH: CABA vs CÓRDOBA (2016-2025)")
print("=" * 80)

# ==============================================================================
# 1. CARGA Y CONSOLIDACIÓN DE DATOS
# ==============================================================================
print("\n1. CARGANDO MICRODATOS EPH...")
print("-" * 80)

# Buscar todos los archivos en eph/raw/
archivos = glob('eph/raw/*.xlsx') + glob('eph/raw/*.xls')
print(f"✓ Archivos encontrados: {len(archivos)}")

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

# ==============================================================================
# 2. CARGA Y MERGE CON IPC
# ==============================================================================
print("\n2. CARGANDO Y APLICANDO IPC...")
print("-" * 80)

# Cargar IPC limpio
ipc = pd.read_csv('ipc_trimestral_limpio.csv')
print(f"✓ IPC cargado: {len(ipc)} registros")

# Verificar que IPC_VAL es numérico
if ipc['IPC_VAL'].dtype == 'object':
    ipc['IPC_VAL'] = ipc['IPC_VAL'].astype(str).str.replace(',', '.').astype(float)
    print(f"✓ IPC_VAL convertido a numérico")

# Mapeo de región EPH a nombres de IPC
# REGION en EPH: 1=GBA, 40=NOA, 41=NEA, 42=Cuyo, 43=Pampeana, 44=Patagonia
df['REGION_IPC'] = df['REGION'].map({
    1: 'GBA',
    40: 'Noroeste',
    41: 'Noreste', 
    42: 'Cuyo',
    43: 'Pampeana',
    44: 'Patagonia'
})

# Verificar mapeo
print(f"✓ Regiones mapeadas:")
print(df.groupby(['REGION', 'REGION_IPC']).size())

# Merge con IPC
df = df.merge(ipc[['ANO4', 'TRIMESTRE', 'REGION', 'IPC_VAL']], 
              left_on=['ANO4', 'TRIMESTRE', 'REGION_IPC'], 
              right_on=['ANO4', 'TRIMESTRE', 'REGION'], 
              how='left', suffixes=('', '_ipc'))

# Verificar merge
registros_sin_ipc = df['IPC_VAL'].isna().sum()
if registros_sin_ipc > 0:
    print(f"⚠ Advertencia: {registros_sin_ipc:,} registros sin IPC")
else:
    print(f"✓ Todos los registros tienen IPC")

# Deflactar ingresos (base último trimestre disponible = 100 del período más reciente)
ipc_base = df['IPC_VAL'].max()
df['P21_REAL'] = np.where(df['IPC_VAL'].notna() & (df['P21'] > 0),
                          (df['P21'] / df['IPC_VAL']) * ipc_base,
                          df['P21'])
df['ITF_REAL'] = np.where(df['IPC_VAL'].notna() & (df['ITF'] > 0),
                          (df['ITF'] / df['IPC_VAL']) * ipc_base,
                          df['ITF'])

print(f"✓ Ingresos deflactados (base IPC = {ipc_base:.2f})")

# ==============================================================================
# 3. CÁLCULO DE TASAS E INDICADORES
# ==============================================================================
print("\n3. CALCULANDO INDICADORES LABORALES...")
print("-" * 80)

# Función para calcular tasas
def calcular_tasas(df_grupo):
    # Población total
    pob_total = df_grupo['PONDERA'].sum()
    
    # PEA (Ocupados + Desocupados)
    pea = df_grupo[df_grupo['ESTADO'].isin([1, 2])]['PONDERA'].sum()
    
    # Ocupados
    ocupados = df_grupo[df_grupo['ESTADO'] == 1]['PONDERA'].sum()
    
    # Desocupados
    desocupados = df_grupo[df_grupo['ESTADO'] == 2]['PONDERA'].sum()
    
    # Tasas
    tasa_actividad = (pea / pob_total) * 100 if pob_total > 0 else 0
    tasa_empleo = (ocupados / pob_total) * 100 if pob_total > 0 else 0
    tasa_desocupacion = (desocupados / pea) * 100 if pea > 0 else 0
    
    return pd.Series({
        'tasa_actividad': tasa_actividad,
        'tasa_empleo': tasa_empleo,
        'tasa_desocupacion': tasa_desocupacion,
        'pob_total': pob_total,
        'pea': pea,
        'ocupados': ocupados,
        'desocupados': desocupados
    })

# Calcular por trimestre y aglomerado
tasas = df.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO_NOMBRE']).apply(calcular_tasas).reset_index()
tasas['PERIODO'] = tasas['ANO4'].astype(str) + '-T' + tasas['TRIMESTRE'].astype(str)

print("✓ Tasas calculadas por trimestre")

# ==============================================================================
# 4. ANÁLISIS UNIVARIADO - INGRESOS
# ==============================================================================
print("\n4. ANÁLISIS UNIVARIADO - INGRESOS...")
print("-" * 80)

# Solo ocupados con ingresos positivos
df_ingresos = df[(df['ESTADO'] == 1) & (df['P21_REAL'] > 0)].copy()

# Estadísticas descriptivas por período
ingresos_stats = df_ingresos.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO_NOMBRE'])['P21_REAL'].agg([
    ('media', 'mean'),
    ('mediana', 'median'),
    ('p10', lambda x: np.percentile(x, 10)),
    ('p25', lambda x: np.percentile(x, 25)),
    ('p75', lambda x: np.percentile(x, 75)),
    ('p90', lambda x: np.percentile(x, 90)),
    ('desvio', 'std')
]).reset_index()

ingresos_stats['PERIODO'] = ingresos_stats['ANO4'].astype(str) + '-T' + ingresos_stats['TRIMESTRE'].astype(str)

print("✓ Estadísticas de ingresos calculadas")

# ==============================================================================
# 5. ANÁLISIS MULTIVARIADO
# ==============================================================================
print("\n5. ANÁLISIS MULTIVARIADO...")
print("-" * 80)

# Por sexo
tasas_sexo = df.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO_NOMBRE', 'CH04']).apply(calcular_tasas).reset_index()
tasas_sexo['SEXO'] = tasas_sexo['CH04'].map({1: 'Varón', 2: 'Mujer'})
tasas_sexo['PERIODO'] = tasas_sexo['ANO4'].astype(str) + '-T' + tasas_sexo['TRIMESTRE'].astype(str)

# Por nivel educativo
tasas_educacion = df.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO_NOMBRE', 'NIVEL_ED']).apply(calcular_tasas).reset_index()
tasas_educacion['PERIODO'] = tasas_educacion['ANO4'].astype(str) + '-T' + tasas_educacion['TRIMESTRE'].astype(str)

# Ingresos por sexo
ingresos_sexo = df_ingresos.groupby(['ANO4', 'TRIMESTRE', 'AGLOMERADO_NOMBRE', 'CH04'])['P21_REAL'].agg([
    ('media', 'mean'),
    ('mediana', 'median')
]).reset_index()
ingresos_sexo['SEXO'] = ingresos_sexo['CH04'].map({1: 'Varón', 2: 'Mujer'})
ingresos_sexo['PERIODO'] = ingresos_sexo['ANO4'].astype(str) + '-T' + ingresos_sexo['TRIMESTRE'].astype(str)

print("✓ Análisis multivariado completado")

# ==============================================================================
# 6. VISUALIZACIONES
# ==============================================================================
print("\n6. GENERANDO VISUALIZACIONES...")
print("-" * 80)

# Configurar subplots
fig = plt.figure(figsize=(20, 24))

# 6.1 Tasa de Desocupación
ax1 = plt.subplot(4, 2, 1)
for aglo in ['CABA', 'Córdoba']:
    data = tasas[tasas['AGLOMERADO_NOMBRE'] == aglo]
    ax1.plot(range(len(data)), data['tasa_desocupacion'], marker='o', label=aglo, linewidth=2)
ax1.set_title('Evolución de la Tasa de Desocupación', fontsize=14, fontweight='bold')
ax1.set_xlabel('Período')
ax1.set_ylabel('Tasa (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, len(data), 4))
ax1.set_xticklabels(data['PERIODO'].iloc[::4], rotation=45)

# 6.2 Tasa de Empleo
ax2 = plt.subplot(4, 2, 2)
for aglo in ['CABA', 'Córdoba']:
    data = tasas[tasas['AGLOMERADO_NOMBRE'] == aglo]
    ax2.plot(range(len(data)), data['tasa_empleo'], marker='o', label=aglo, linewidth=2)
ax2.set_title('Evolución de la Tasa de Empleo', fontsize=14, fontweight='bold')
ax2.set_xlabel('Período')
ax2.set_ylabel('Tasa (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, len(data), 4))
ax2.set_xticklabels(data['PERIODO'].iloc[::4], rotation=45)

# 6.3 Tasa de Actividad
ax3 = plt.subplot(4, 2, 3)
for aglo in ['CABA', 'Córdoba']:
    data = tasas[tasas['AGLOMERADO_NOMBRE'] == aglo]
    ax3.plot(range(len(data)), data['tasa_actividad'], marker='o', label=aglo, linewidth=2)
ax3.set_title('Evolución de la Tasa de Actividad', fontsize=14, fontweight='bold')
ax3.set_xlabel('Período')
ax3.set_ylabel('Tasa (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, len(data), 4))
ax3.set_xticklabels(data['PERIODO'].iloc[::4], rotation=45)

# 6.4 Ingreso Real Medio
ax4 = plt.subplot(4, 2, 4)
for aglo in ['CABA', 'Córdoba']:
    data = ingresos_stats[ingresos_stats['AGLOMERADO_NOMBRE'] == aglo]
    ax4.plot(range(len(data)), data['media'], marker='o', label=aglo, linewidth=2)
ax4.set_title('Evolución del Ingreso Real Medio', fontsize=14, fontweight='bold')
ax4.set_xlabel('Período')
ax4.set_ylabel('Ingreso Real ($)')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, len(data), 4))
ax4.set_xticklabels(data['PERIODO'].iloc[::4], rotation=45)

# 6.5 Desocupación por Sexo - CABA
ax5 = plt.subplot(4, 2, 5)
for sexo in ['Varón', 'Mujer']:
    data = tasas_sexo[(tasas_sexo['AGLOMERADO_NOMBRE'] == 'CABA') & (tasas_sexo['SEXO'] == sexo)]
    ax5.plot(range(len(data)), data['tasa_desocupacion'], marker='o', label=sexo, linewidth=2)
ax5.set_title('Tasa de Desocupación por Sexo - CABA', fontsize=14, fontweight='bold')
ax5.set_xlabel('Período')
ax5.set_ylabel('Tasa (%)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6.6 Desocupación por Sexo - Córdoba
ax6 = plt.subplot(4, 2, 6)
for sexo in ['Varón', 'Mujer']:
    data = tasas_sexo[(tasas_sexo['AGLOMERADO_NOMBRE'] == 'Córdoba') & (tasas_sexo['SEXO'] == sexo)]
    ax6.plot(range(len(data)), data['tasa_desocupacion'], marker='o', label=sexo, linewidth=2)
ax6.set_title('Tasa de Desocupación por Sexo - Córdoba', fontsize=14, fontweight='bold')
ax6.set_xlabel('Período')
ax6.set_ylabel('Tasa (%)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 6.7 Ingresos por Sexo - CABA
ax7 = plt.subplot(4, 2, 7)
for sexo in ['Varón', 'Mujer']:
    data = ingresos_sexo[(ingresos_sexo['AGLOMERADO_NOMBRE'] == 'CABA') & (ingresos_sexo['SEXO'] == sexo)]
    ax7.plot(range(len(data)), data['media'], marker='o', label=sexo, linewidth=2)
ax7.set_title('Ingreso Real Medio por Sexo - CABA', fontsize=14, fontweight='bold')
ax7.set_xlabel('Período')
ax7.set_ylabel('Ingreso Real ($)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 6.8 Ingresos por Sexo - Córdoba
ax8 = plt.subplot(4, 2, 8)
for sexo in ['Varón', 'Mujer']:
    data = ingresos_sexo[(ingresos_sexo['AGLOMERADO_NOMBRE'] == 'Córdoba') & (ingresos_sexo['SEXO'] == sexo)]
    ax8.plot(range(len(data)), data['media'], marker='o', label=sexo, linewidth=2)
ax8.set_title('Ingreso Real Medio por Sexo - Córdoba', fontsize=14, fontweight='bold')
ax8.set_xlabel('Período')
ax8.set_ylabel('Ingreso Real ($)')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_eph_caba_cordoba.png', dpi=300, bbox_inches='tight')
print("✓ Gráficos guardados en 'analisis_eph_caba_cordoba.png'")

# ==============================================================================
# 7. EXPORTAR TABLAS
# ==============================================================================
print("\n7. EXPORTANDO TABLAS...")
print("-" * 80)

# Crear Excel con múltiples hojas
with pd.ExcelWriter('resultados_eph_caba_cordoba.xlsx', engine='openpyxl') as writer:
    tasas.to_excel(writer, sheet_name='Tasas_Generales', index=False)
    ingresos_stats.to_excel(writer, sheet_name='Ingresos_Estadisticas', index=False)
    tasas_sexo.to_excel(writer, sheet_name='Tasas_por_Sexo', index=False)
    ingresos_sexo.to_excel(writer, sheet_name='Ingresos_por_Sexo', index=False)
    tasas_educacion.to_excel(writer, sheet_name='Tasas_por_Educacion', index=False)

print("✓ Tablas exportadas a 'resultados_eph_caba_cordoba.xlsx'")

# ==============================================================================
# 8. RESUMEN ESTADÍSTICO
# ==============================================================================
print("\n" + "=" * 80)
print("RESUMEN ESTADÍSTICO FINAL")
print("=" * 80)

# Último trimestre disponible
ultimo_periodo = tasas.groupby('AGLOMERADO_NOMBRE').tail(1)

print("\nÚLTIMO PERÍODO DISPONIBLE:")
print(ultimo_periodo[['AGLOMERADO_NOMBRE', 'PERIODO', 'tasa_desocupacion', 'tasa_empleo', 'tasa_actividad']])

print("\n✓ ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 80)