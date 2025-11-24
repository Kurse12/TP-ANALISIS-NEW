import pandas as pd
import numpy as np

print("=" * 80)
print("EXPORTANDO ESTADÍSTICAS A TXT")
print("=" * 80)

# Abrir el archivo Excel con todas las hojas
excel_file = 'resultados_eph_caba_cordoba.xlsx'
xls = pd.ExcelFile(excel_file)

print(f"\n✓ Archivo cargado: {excel_file}")
print(f"✓ Hojas encontradas: {len(xls.sheet_names)}")

# Crear archivo de texto
with open('estadisticas_eph_resumen.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESUMEN DE ESTADÍSTICAS EPH - CABA vs CÓRDOBA (2016-2025)\n")
    f.write("=" * 80 + "\n\n")
    
    # =========================================================================
    # 1. TASAS GENERALES
    # =========================================================================
    df_tasas = pd.read_excel(excel_file, sheet_name='Tasas_Generales')
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("1. TASAS LABORALES - EVOLUCIÓN GENERAL\n")
    f.write("=" * 80 + "\n\n")
    
    for aglo in ['CABA', 'Córdoba']:
        df_aglo = df_tasas[df_tasas['AGLOMERADO_NOMBRE'] == aglo].sort_values(['ANO4', 'TRIMESTRE'])
        
        f.write(f"\n{'─' * 80}\n")
        f.write(f"{aglo}\n")
        f.write(f"{'─' * 80}\n\n")
        
        # Primeros y últimos 4 trimestres
        f.write("PRIMEROS 4 TRIMESTRES (2016):\n")
        f.write("-" * 80 + "\n")
        primeros = df_aglo.head(4)
        for _, row in primeros.iterrows():
            f.write(f"{row['PERIODO']:8} | Desocup: {row['tasa_desocupacion']:5.2f}% | "
                   f"Empleo: {row['tasa_empleo']:5.2f}% | "
                   f"Actividad: {row['tasa_actividad']:5.2f}%\n")
        
        f.write("\nÚLTIMOS 4 TRIMESTRES:\n")
        f.write("-" * 80 + "\n")
        ultimos = df_aglo.tail(4)
        for _, row in ultimos.iterrows():
            f.write(f"{row['PERIODO']:8} | Desocup: {row['tasa_desocupacion']:5.2f}% | "
                   f"Empleo: {row['tasa_empleo']:5.2f}% | "
                   f"Actividad: {row['tasa_actividad']:5.2f}%\n")
        
        # Estadísticas resumidas
        f.write("\nESTADÍSTICAS PERÍODO COMPLETO:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Desocupación - Promedio: {df_aglo['tasa_desocupacion'].mean():.2f}% | "
               f"Mín: {df_aglo['tasa_desocupacion'].min():.2f}% | "
               f"Máx: {df_aglo['tasa_desocupacion'].max():.2f}%\n")
        f.write(f"Empleo       - Promedio: {df_aglo['tasa_empleo'].mean():.2f}% | "
               f"Mín: {df_aglo['tasa_empleo'].min():.2f}% | "
               f"Máx: {df_aglo['tasa_empleo'].max():.2f}%\n")
        f.write(f"Actividad    - Promedio: {df_aglo['tasa_actividad'].mean():.2f}% | "
               f"Mín: {df_aglo['tasa_actividad'].min():.2f}% | "
               f"Máx: {df_aglo['tasa_actividad'].max():.2f}%\n")
        
        # Identificar picos de desocupación
        max_desocup = df_aglo.loc[df_aglo['tasa_desocupacion'].idxmax()]
        min_desocup = df_aglo.loc[df_aglo['tasa_desocupacion'].idxmin()]
        f.write(f"\nPico de DESOCUPACIÓN: {max_desocup['PERIODO']} ({max_desocup['tasa_desocupacion']:.2f}%)\n")
        f.write(f"Mínimo de DESOCUPACIÓN: {min_desocup['PERIODO']} ({min_desocup['tasa_desocupacion']:.2f}%)\n")
    
    # Comparación directa
    f.write("\n" + "=" * 80 + "\n")
    f.write("COMPARACIÓN CABA vs CÓRDOBA (Promedios del período)\n")
    f.write("=" * 80 + "\n")
    
    comparacion = df_tasas.groupby('AGLOMERADO_NOMBRE')[['tasa_desocupacion', 'tasa_empleo', 'tasa_actividad']].mean()
    f.write(f"\nTasa de Desocupación:\n")
    f.write(f"  CABA:    {comparacion.loc['CABA', 'tasa_desocupacion']:.2f}%\n")
    f.write(f"  Córdoba: {comparacion.loc['Córdoba', 'tasa_desocupacion']:.2f}%\n")
    f.write(f"  Diferencia: {abs(comparacion.loc['CABA', 'tasa_desocupacion'] - comparacion.loc['Córdoba', 'tasa_desocupacion']):.2f} p.p.\n")
    
    f.write(f"\nTasa de Empleo:\n")
    f.write(f"  CABA:    {comparacion.loc['CABA', 'tasa_empleo']:.2f}%\n")
    f.write(f"  Córdoba: {comparacion.loc['Córdoba', 'tasa_empleo']:.2f}%\n")
    f.write(f"  Diferencia: {abs(comparacion.loc['CABA', 'tasa_empleo'] - comparacion.loc['Córdoba', 'tasa_empleo']):.2f} p.p.\n")
    
    # =========================================================================
    # 2. INGRESOS
    # =========================================================================
    df_ingresos = pd.read_excel(excel_file, sheet_name='Ingresos_Estadisticas')
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("2. INGRESOS REALES (ajustados por inflación)\n")
    f.write("=" * 80 + "\n\n")
    
    for aglo in ['CABA', 'Córdoba']:
        df_aglo = df_ingresos[df_ingresos['AGLOMERADO_NOMBRE'] == aglo].sort_values(['ANO4', 'TRIMESTRE'])
        
        f.write(f"\n{'─' * 80}\n")
        f.write(f"{aglo}\n")
        f.write(f"{'─' * 80}\n\n")
        
        # Primeros trimestres
        f.write("PRIMEROS 4 TRIMESTRES (2016):\n")
        f.write("-" * 80 + "\n")
        primeros = df_aglo.head(4)
        for _, row in primeros.iterrows():
            f.write(f"{row['PERIODO']:8} | Media: ${row['media']:,.0f} | "
                   f"Mediana: ${row['mediana']:,.0f} | "
                   f"P90/P10: {row['p90']/row['p10']:.2f}\n")
        
        # Últimos trimestres
        f.write("\nÚLTIMOS 4 TRIMESTRES:\n")
        f.write("-" * 80 + "\n")
        ultimos = df_aglo.tail(4)
        for _, row in ultimos.iterrows():
            f.write(f"{row['PERIODO']:8} | Media: ${row['media']:,.0f} | "
                   f"Mediana: ${row['mediana']:,.0f} | "
                   f"P90/P10: {row['p90']/row['p10']:.2f}\n")
        
        # Estadísticas del período
        f.write("\nESTADÍSTICAS PERÍODO COMPLETO:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Ingreso MEDIO    - Promedio: ${df_aglo['media'].mean():,.0f} | "
               f"Mín: ${df_aglo['media'].min():,.0f} | "
               f"Máx: ${df_aglo['media'].max():,.0f}\n")
        f.write(f"Ingreso MEDIANA  - Promedio: ${df_aglo['mediana'].mean():,.0f} | "
               f"Mín: ${df_aglo['mediana'].min():,.0f} | "
               f"Máx: ${df_aglo['mediana'].max():,.0f}\n")
        
        # Desigualdad
        desigualdad_promedio = (df_aglo['p90'] / df_aglo['p10']).mean()
        f.write(f"\nDESIGUALDAD (P90/P10 promedio): {desigualdad_promedio:.2f}\n")
        f.write(f"  → El 10% que más gana, gana {desigualdad_promedio:.1f} veces más que el 10% que menos gana\n")
        
        # Evolución (comparar primer vs último año disponible)
        primer_año = df_aglo[df_aglo['ANO4'] == df_aglo['ANO4'].min()]['media'].mean()
        ultimo_año = df_aglo[df_aglo['ANO4'] == df_aglo['ANO4'].max()]['media'].mean()
        variacion = ((ultimo_año - primer_año) / primer_año) * 100
        f.write(f"\nVARIACIÓN INGRESO REAL:\n")
        f.write(f"  Año {df_aglo['ANO4'].min()}: ${primer_año:,.0f}\n")
        f.write(f"  Año {df_aglo['ANO4'].max()}: ${ultimo_año:,.0f}\n")
        f.write(f"  Variación: {variacion:+.1f}%\n")
    
    # Comparación CABA vs Córdoba
    f.write("\n" + "=" * 80 + "\n")
    f.write("COMPARACIÓN INGRESOS CABA vs CÓRDOBA\n")
    f.write("=" * 80 + "\n")
    
    comp_ingresos = df_ingresos.groupby('AGLOMERADO_NOMBRE')[['media', 'mediana']].mean()
    brecha = ((comp_ingresos.loc['CABA', 'media'] / comp_ingresos.loc['Córdoba', 'media']) - 1) * 100
    
    f.write(f"\nIngreso Medio:\n")
    f.write(f"  CABA:    ${comp_ingresos.loc['CABA', 'media']:,.0f}\n")
    f.write(f"  Córdoba: ${comp_ingresos.loc['Córdoba', 'media']:,.0f}\n")
    f.write(f"  CABA gana {brecha:+.1f}% más que Córdoba\n")
    
    # =========================================================================
    # 3. ANÁLISIS POR SEXO
    # =========================================================================
    df_sexo = pd.read_excel(excel_file, sheet_name='Tasas_por_Sexo')
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("3. ANÁLISIS POR SEXO\n")
    f.write("=" * 80 + "\n\n")
    
    for aglo in ['CABA', 'Córdoba']:
        f.write(f"\n{'─' * 80}\n")
        f.write(f"{aglo}\n")
        f.write(f"{'─' * 80}\n\n")
        
        df_aglo_sexo = df_sexo[df_sexo['AGLOMERADO_NOMBRE'] == aglo]
        
        for sexo in ['Varón', 'Mujer']:
            df_s = df_aglo_sexo[df_aglo_sexo['SEXO'] == sexo]
            f.write(f"{sexo}:\n")
            f.write(f"  Desocupación promedio: {df_s['tasa_desocupacion'].mean():.2f}%\n")
            f.write(f"  Empleo promedio:       {df_s['tasa_empleo'].mean():.2f}%\n")
            f.write(f"  Actividad promedio:    {df_s['tasa_actividad'].mean():.2f}%\n\n")
        
        # Brechas de género
        desocup_varon = df_aglo_sexo[df_aglo_sexo['SEXO'] == 'Varón']['tasa_desocupacion'].mean()
        desocup_mujer = df_aglo_sexo[df_aglo_sexo['SEXO'] == 'Mujer']['tasa_desocupacion'].mean()
        brecha_desocup = desocup_mujer - desocup_varon
        
        f.write(f"BRECHA DE GÉNERO (Mujer - Varón):\n")
        f.write(f"  Desocupación: {brecha_desocup:+.2f} p.p. ")
        if brecha_desocup > 0:
            f.write(f"(las mujeres tienen {brecha_desocup:.2f} p.p. más de desocupación)\n")
        else:
            f.write(f"(los varones tienen {abs(brecha_desocup):.2f} p.p. más de desocupación)\n")
    
    # =========================================================================
    # 4. INGRESOS POR SEXO
    # =========================================================================
    df_ing_sexo = pd.read_excel(excel_file, sheet_name='Ingresos_por_Sexo')
    
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("4. BRECHA SALARIAL DE GÉNERO\n")
    f.write("=" * 80 + "\n\n")
    
    for aglo in ['CABA', 'Córdoba']:
        df_aglo_ing = df_ing_sexo[df_ing_sexo['AGLOMERADO_NOMBRE'] == aglo]
        
        f.write(f"\n{'─' * 80}\n")
        f.write(f"{aglo}\n")
        f.write(f"{'─' * 80}\n\n")
        
        ing_varon = df_aglo_ing[df_aglo_ing['SEXO'] == 'Varón']['media'].mean()
        ing_mujer = df_aglo_ing[df_aglo_ing['SEXO'] == 'Mujer']['media'].mean()
        brecha_salarial = ((ing_varon / ing_mujer) - 1) * 100
        
        f.write(f"Ingreso medio Varón:  ${ing_varon:,.0f}\n")
        f.write(f"Ingreso medio Mujer:  ${ing_mujer:,.0f}\n")
        f.write(f"Brecha salarial: {brecha_salarial:.1f}% ")
        f.write(f"(los varones ganan {brecha_salarial:.1f}% más que las mujeres)\n")

print("\n✓ Archivo creado: estadisticas_eph_resumen.txt")
print("\n✓ Este archivo contiene:")
print("  - Evolución de tasas laborales")
print("  - Estadísticas de ingresos reales")
print("  - Comparación CABA vs Córdoba")
print("  - Análisis por sexo y brechas de género")
print("\n" + "=" * 80)