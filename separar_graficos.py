from PIL import Image
import os

print("=" * 80)
print("SEPARANDO GRÁFICOS INDIVIDUALES")
print("=" * 80)

# Cargar la imagen completa
img = Image.open('analisis_eph_caba_cordoba.png')
width, height = img.size

print(f"\n✓ Imagen cargada: {width}x{height} píxeles")

# Crear carpeta para los gráficos individuales
if not os.path.exists('graficos_individuales'):
    os.makedirs('graficos_individuales')
    print("✓ Carpeta 'graficos_individuales' creada")

# La imagen tiene 4 filas x 2 columnas = 8 gráficos
filas = 4
columnas = 2

# Calcular dimensiones de cada gráfico
ancho_grafico = width // columnas
alto_grafico = height // filas

# Nombres descriptivos para cada gráfico
nombres = [
    "01_tasa_desocupacion",
    "02_tasa_empleo",
    "03_tasa_actividad",
    "04_ingreso_real_medio",
    "05_desocupacion_sexo_caba",
    "06_desocupacion_sexo_cordoba",
    "07_ingresos_sexo_caba",
    "08_ingresos_sexo_cordoba"
]

print(f"\nSeparando en {filas}x{columnas} = {len(nombres)} gráficos:\n")

# Extraer cada gráfico
contador = 0
for fila in range(filas):
    for col in range(columnas):
        # Calcular coordenadas
        left = col * ancho_grafico
        top = fila * alto_grafico
        right = left + ancho_grafico
        bottom = top + alto_grafico
        
        # Recortar
        grafico = img.crop((left, top, right, bottom))
        
        # Guardar
        nombre_archivo = f"graficos_individuales/{nombres[contador]}.png"
        grafico.save(nombre_archivo, dpi=(300, 300))
        
        print(f"✓ {nombres[contador]}.png")
        contador += 1

print(f"\n✓ {len(nombres)} gráficos guardados en 'graficos_individuales/'")
print("=" * 80)