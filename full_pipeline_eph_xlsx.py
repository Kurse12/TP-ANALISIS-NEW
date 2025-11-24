# full_pipeline_eph_xlsx.py
"""
Pipeline EPH (xlsx) 2016-2025 -> limpia, merge IPC, deflacta, calcula tasas ponderadas,
imputa ingresos (RandomForest), exporta tablas + figuras.
Diseñado para comparar GBA (Buenos Aires) vs Pampeana (Córdoba).
"""

import os, glob, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.options.mode.chained_assignment = None

# ---------------- CONFIG ----------------
RAW_DIR = "eph/raw"      # carpeta con los .xlsx
IPC_FILE = "ipc_trimestral.csv"
OUTPUT_DIR = "output_tp"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# posibles nombres de columnas (heurísticos)
POSSIBLE_INCOME_COLS = ["P21","P47T","ingreso","ingresos","P47"]
POSSIBLE_PON_COLS = ["PONDIIO","pondiio","PONDI","factor","fex_c","fex"]
POSSIBLE_REGION_COLS = ["REGION","REGHOG","region","region_hogar","aglo","AGLOMERADO"]
POSSIBLE_EDU_COLS = ["NIVEL_ED","nivel_ed","estudios","educ"]
POSSIBLE_SEX_COLS = ["CH04","sexo","SEXO","sexo_persona"]
POSSIBLE_AGE_COLS = ["CH06","edad","EDAD"]

# helper: weighted stats
def wmean(x, w):
    mask = (~x.isna()) & (~w.isna())
    if mask.sum() == 0: return np.nan
    return (x[mask]*w[mask]).sum() / w[mask].sum()

def wmedian(values, weights):
    # simple approximate weighted median
    v = np.array(values)
    w = np.array(weights)
    mask = ~np.isnan(v)
    if mask.sum() == 0: return np.nan
    v = v[mask]; w = w[mask]
    sorter = np.argsort(v)
    v = v[sorter]; w = w[sorter]
    cumsum = np.cumsum(w)
    cutoff = cumsum[-1]/2.0
    idx = np.searchsorted(cumsum, cutoff)
    return float(v[min(idx, len(v)-1)])

def wquantile(values, weights, q):
    v = np.array(values); w = np.array(weights)
    mask = ~np.isnan(v)
    if mask.sum()==0: return np.nan
    v = v[mask]; w = w[mask]
    sorter = np.argsort(v)
    v = v[sorter]; w = w[sorter]
    cumsum = np.cumsum(w)
    cutoff = q * cumsum[-1]
    idx = np.searchsorted(cumsum, cutoff)
    return float(v[min(idx, len(v)-1)])

# ---------------- 1) Cargar IPC (preparado previamente) ----------------
print("Cargando IPC desde:", IPC_FILE)
ipc = pd.read_csv(IPC_FILE, sep=";", encoding="latin1") if IPC_FILE.endswith(".csv") else pd.read_csv(IPC_FILE, encoding="latin1")
# detectar columnas
ipc_cols = [c.strip() for c in ipc.columns]
ipc.columns = ipc_cols
if "Periodo" not in ipc.columns:
    # intentar minúscula
    if "periodo" in ipc.columns:
        ipc.rename(columns={"periodo":"Periodo"}, inplace=True)
# convertir Periodo (YYYYMM) -> fecha trimestral
ipc["Periodo"] = ipc["Periodo"].astype(str)
ipc["fecha"] = pd.to_datetime(ipc["Periodo"], format="%Y%m", errors="coerce")
# quedarnos solo nivel general y regiones GBA/Pampeana
if "Descripcion" in ipc.columns:
    ipc = ipc[ipc["Descripcion"].str.upper().str.contains("NIVEL GENERAL", na=False)]
# normalizar Region col name
region_col = [c for c in ipc.columns if c.lower()=="region"]
region_col = region_col[0] if region_col else None
if region_col is None:
    raise ValueError("IPC: no encuentro columna 'Region' en ipc_trimestral.csv")
ipc = ipc[ipc[region_col].isin(["GBA","Pampeana","GBA ","Pampeana "])]
ipc[region_col] = ipc[region_col].str.strip()
# asignar ANO4/TRIMESTRE
ipc['ANO4'] = ipc['fecha'].dt.year
ipc['TRIMESTRE'] = ((ipc['fecha'].dt.month-1)//3 + 1)
# indice col
possible_ind = [c for c in ipc.columns if 'indice' in c.lower()]
if len(possible_ind)==0:
    raise ValueError("IPC: no encuentro columna de Indice IPC")
ipc = ipc.rename(columns={possible_ind[0]:'IPC_VAL'})

print("IPC listo: períodos", ipc['ANO4'].min(), "->", ipc['ANO4'].max())

# ---------------- 2) Leer todos los xlsx EPH ----------------
print("\nBuscando archivos .xlsx en", RAW_DIR)
files = glob.glob(os.path.join(RAW_DIR, "*.xlsx")) + glob.glob(os.path.join(RAW_DIR,"*.xls"))
files = sorted(files)
if len(files)==0:
    raise FileNotFoundError(f"No encontré archivos .xlsx/.xls en {RAW_DIR}")

all_dfs = []
for f in files:
    print("Leyendo:", f)
    try:
        tmp = pd.read_excel(f, engine="openpyxl")
    except Exception:
        tmp = pd.read_excel(f)
    # guardar nombre archivo para trazabilidad
    tmp['_source_file'] = os.path.basename(f)
    # intentar inferir ANO4/TRIMESTRE desde el nombre
    base = os.path.basename(f)
    # buscar 4 dígitos de año y T1..T4 o _T1_
    import re
    m_year = re.search(r"(19|20)\d{2}", base)
    year = int(m_year.group(0)) if m_year else None
    m_tri = re.search(r"[Tt]([1-4])", base)
    tri = int(m_tri.group(1)) if m_tri else None
    if year is not None:
        tmp['ANO4'] = year
    if tri is not None:
        tmp['TRIMESTRE'] = tri
    all_dfs.append(tmp)

eph = pd.concat(all_dfs, ignore_index=True, sort=False)
print("EPH concatenado. filas:", len(eph), " columnas:", len(eph.columns))

# ---------------- 3) Detectar columnas clave en EPH ----------------
cols = [c for c in eph.columns]
cols_lower = [c.lower() for c in cols]

# ingreso
income_col = None
for cand in POSSIBLE_INCOME_COLS:
    for c in cols:
        if c.upper()==cand.upper() or cand.lower() in c.lower():
            income_col = c
            break
    if income_col: break

# ponderador
pon_col = None
for cand in POSSIBLE_PON_COLS:
    for c in cols:
        if cand.lower() in c.lower():
            pon_col = c
            break
    if pon_col: break

# region
region_col = None
for cand in POSSIBLE_REGION_COLS:
    for c in cols:
        if cand.lower() == c.lower() or cand.lower() in c.lower():
            region_col = c
            break
    if region_col: break

# sexo, edad, estado (condicion laboral)
sex_col = None
for cand in POSSIBLE_SEX_COLS:
    for c in cols:
        if cand.lower() in c.lower():
            sex_col = c; break
    if sex_col: break

age_col = None
for cand in POSSIBLE_AGE_COLS:
    for c in cols:
        if cand.lower() in c.lower():
            age_col = c; break
    if age_col: break

estado_col = None
for c in cols:
    if c.upper() in ["ESTADO","ESTADO_ACT","PP04","CONDICION","ESTADO_OCUP"]:
        estado_col = c; break
# si no está ESTADO, intentar por nombres comunes
if estado_col is None:
    for c in cols:
        if 'estado' in c.lower() or 'condicion' in c.lower():
            estado_col = c; break

print("\nColumnas detectadas:")
print("Ingreso:", income_col)
print("Ponderador:", pon_col)
print("Region:", region_col)
print("Sexo:", sex_col)
print("Edad:", age_col)
print("Estado laboral:", estado_col)

# validar detecciones
if income_col is None:
    print("⚠ No detecté columna de ingreso automáticamente. Editá POSSIBLE_INCOME_COLS o ponela manualmente.")
else:
    eph['P21'] = pd.to_numeric(eph[income_col], errors='coerce')

if pon_col is None:
    raise ValueError("No encontré columna de ponderador. Buscá en tus xlsx la columna apropiada (PONDIIO, factor, etc).")
else:
    eph['PONDIIO'] = pd.to_numeric(eph[pon_col], errors='coerce')

if region_col is None:
    raise ValueError("No encontré columna de region en EPH. Revisa nombres de columnas en tus xlsx.")
else:
    eph['REGION_RAW'] = eph[region_col]

# edad y sexo fallback
if age_col:
    eph['edad'] = pd.to_numeric(eph[age_col], errors='coerce')
if sex_col:
    eph['sexo'] = eph[sex_col].astype(str)

# ESTADO -> recodificar a Ocupado/Desocupado/Inactivo
def recode_estado(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    # heurístico: 1 ocupado, 2 desocupado, 3 inactivo (formato INDEC)
    if s in ['1','1.0', 'Ocupado', 'OCUPADO', 'ocupado']: return 'Ocupado'
    if s in ['2','2.0', 'Desocupado', 'DESOCUPADO', 'desocupado']: return 'Desocupado'
    if s in ['3','3.0', 'Inactivo', 'INACTIVO', 'inactivo']: return 'Inactivo'
    # intentar buscar palabras
    if 'ocu' in s.lower(): return 'Ocupado'
    if 'des' in s.lower(): return 'Desocupado'
    if 'inac' in s.lower(): return 'Inactivo'
    return np.nan

if estado_col:
    eph['CONDICION'] = eph[estado_col].apply(recode_estado)
else:
    eph['CONDICION'] = np.nan

# ---------------- 4) Filtrar población 14+ y crear PERIODO ----------------
if 'edad' in eph.columns:
    eph = eph[eph['edad'] >= 14].copy()

# si no detectamos ANO4/TRIMESTRE en archivos, intentar extraer desde columnas existentes
if 'ANO4' not in eph.columns or 'TRIMESTRE' not in eph.columns:
    # buscar columnas que parezcan año/trimestre
    for c in eph.columns:
        if 'ano' in c.lower() and '4' in c.lower():
            eph.rename(columns={c:'ANO4'}, inplace=True)
        if 'trimestre' in c.lower() or 'trim' in c.lower():
            eph.rename(columns={c:'TRIMESTRE'}, inplace=True)

# si aún faltan, intentar extraer de filename (ya lo intentamos), sino error
if 'ANO4' not in eph.columns or 'TRIMESTRE' not in eph.columns:
    # intentar extraer de fecha si existe
    possible_date_cols = [c for c in eph.columns if 'fecha' in c.lower() or 'date' in c.lower()]
    if len(possible_date_cols)>0:
        eph['__fecha_tmp'] = pd.to_datetime(eph[possible_date_cols[0]], errors='coerce')
        eph['ANO4'] = eph['__fecha_tmp'].dt.year
        eph['TRIMESTRE'] = ((eph['__fecha_tmp'].dt.month-1)//3 + 1)
    else:
        # si no se puede, lanzar aviso
        raise ValueError("No encontré columnas ANO4/TRIMESTRE ni fecha en los xlsx. Asegurate que los archivos tengan año y trimestre en el nombre o en columnas.")

eph['PERIODO'] = eph['ANO4'].astype(int).astype(str) + 'T' + eph['TRIMESTRE'].astype(int).astype(str)

# ---------------- 5) Mapear regiones a GBA / Pampeana ----------------
# si REGION_RAW es numérica, usamos mapping heurístico; si texto, buscamos palabras
if np.issubdtype(eph['REGION_RAW'].dtype, np.number):
    # heurístico común: 1 = GBA, 2 = Pampeana (puede variar por dataset)
    # Si tus códigos son otros, editá este mapping a conveniencia
    region_map = {1: "GBA", 2: "Pampeana"}
    eph['REGION_NAME'] = eph['REGION_RAW'].map(region_map).fillna(eph['REGION_RAW'])
else:
    eph['REGION_NAME'] = eph['REGION_RAW'].astype(str).str.strip()
    # simplified mapping
    eph.loc[eph['REGION_NAME'].str.contains("GBA|Buenos|Buenos Aires|CABA", na=False, case=False), 'REGION_NAME'] = 'GBA'
    eph.loc[eph['REGION_NAME'].str.contains("Pampeana|Córdoba|Cordoba|Cordoba", na=False, case=False), 'REGION_NAME'] = 'Pampeana'

# quedarnos solo GBA y Pampeana
eph = eph[eph['REGION_NAME'].isin(['GBA','Pampeana'])].copy()
print("Filtrado por regiones: filas resultantes:", len(eph))
if eph.empty:
    raise ValueError("No quedaron filas después del filtro por GBA/Pampeana. Revisa mapping de regiones.")

# ---------------- 6) Merge EPH con IPC para deflactar ingresos ----------------
# preparar tabla IPC con ANO4/TRIMESTRE/IPC_VAL/Region
ipc_small = ipc[['ANO4','TRIMESTRE', region_col, 'IPC_VAL']].copy()
ipc_small = ipc_small.rename(columns={region_col:'REGION_IPC'})
# normalizar REGION_IPC
ipc_small['REGION_IPC'] = ipc_small['REGION_IPC'].str.strip()
# mapear los nombres iguales a GBA/Pampeana
ipc_small['REGION_IPC'] = ipc_small['REGION_IPC'].replace({'GBA':'GBA','Pampeana':'Pampeana'})

# merge por ANO4/TRIMESTRE y región
eph = eph.merge(ipc_small, left_on=['ANO4','TRIMESTRE','REGION_NAME'], right_on=['ANO4','TRIMESTRE','REGION_IPC'], how='left')
# si IPC faltante, intentar merge solo por ANO4/TRIMESTRE con region ANY (fallback)
if eph['IPC_VAL'].isna().any():
    eph = eph.merge(ipc_small[['ANO4','TRIMESTRE','IPC_VAL']].drop_duplicates(), on=['ANO4','TRIMESTRE'], how='left', suffixes=('','_any'))
    eph['IPC_VAL'] = eph['IPC_VAL'].fillna(eph.get('IPC_VAL_any'))
    eph.drop(columns=[c for c in eph.columns if c.endswith('_any')], inplace=True, errors='ignore')

# elegir IPC base (por ejemplo media 2016)
ipc_base_val = ipc_small.loc[ipc_small['ANO4']==2016,'IPC_VAL'].mean()
if np.isnan(ipc_base_val):
    ipc_base_val = ipc_small['IPC_VAL'].max()

eph['ingreso_real'] = eph['P21'] * (ipc_base_val / eph['IPC_VAL'])

# ---------------- 7) Cálculo de indicadores ponderados por PERIODO y REGION_NAME ----------------
def indicadores_por_periodo(df):
    out = []
    for (per, reg), sub in df.groupby(['PERIODO','REGION_NAME']):
        pobl_w = sub['PONDIIO'].sum()
        ocup_w = sub.loc[sub['CONDICION']=='Ocupado','PONDIIO'].sum()
        des_w = sub.loc[sub['CONDICION']=='Desocupado','PONDIIO'].sum()
        pea_w = ocup_w + des_w
        tasa_des = (des_w/pea_w*100) if pea_w>0 else np.nan
        tasa_emp = (ocup_w/pobl_w*100) if pobl_w>0 else np.nan
        tasa_act = (pea_w/pobl_w*100) if pobl_w>0 else np.nan
        ing_prom = wmean(sub['ingreso_real'], sub['PONDIIO'])
        ing_med = wmedian(sub['ingreso_real'], sub['PONDIIO'])
        p25 = wquantile(sub['ingreso_real'], sub['PONDIIO'], 0.25)
        p75 = wquantile(sub['ingreso_real'], sub['PONDIIO'], 0.75)
        out.append({
            'PERIODO': per, 'REGION': reg,
            'pobl_w': pobl_w, 'ocup_w': ocup_w, 'des_w': des_w, 'pea_w': pea_w,
            'tasa_desocupacion': tasa_des, 'tasa_empleo': tasa_emp, 'tasa_actividad': tasa_act,
            'ingreso_promedio_real': ing_prom, 'ingreso_mediana_real': ing_med,
            'ingreso_p25': p25, 'ingreso_p75': p75
        })
    return pd.DataFrame(out)

print("\nCalculando indicadores ponderados...")
ind = indicadores_por_periodo(eph)
ind = ind.sort_values(['PERIODO','REGION'])
ind.to_csv(os.path.join(OUTPUT_DIR,"indicadores_univariado_ponderados.csv"), index=False)
print("Indicadores guardados en:", os.path.join(OUTPUT_DIR,"indicadores_univariado_ponderados.csv"))

# ---------------- 8) Guardar dataset imputado/parcial ----------------
eph.to_parquet(os.path.join(OUTPUT_DIR,"microdatos_eph_processed.parquet"), index=False)
print("Microdatos procesados guardados en parquet.")

# ---------------- 9) Gráficos principales (series) ----------------
plt.figure(figsize=(12,5))
for reg in ind['REGION'].unique():
    sub = ind[ind['REGION']==reg]
    plt.plot(sub['PERIODO'], sub['tasa_desocupacion'], marker='o', label=reg)
plt.xticks(rotation=45)
plt.title("Tasa de desocupación (%) — GBA vs Pampeana")
plt.ylabel("Tasa desocupación (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"tasa_desocupacion.png"))
plt.close()

plt.figure(figsize=(12,5))
for reg in ind['REGION'].unique():
    sub = ind[ind['REGION']==reg]
    plt.plot(sub['PERIODO'], sub['tasa_empleo'], marker='o', label=reg)
plt.xticks(rotation=45)
plt.title("Tasa de empleo (%) — GBA vs Pampeana")
plt.ylabel("Tasa empleo (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"tasa_empleo.png"))
plt.close()

plt.figure(figsize=(12,5))
for reg in ind['REGION'].unique():
    sub = ind[ind['REGION']==reg]
    plt.plot(sub['PERIODO'], sub['ingreso_mediana_real'], marker='o', label=reg)
plt.xticks(rotation=45)
plt.title("Ingreso mediana real — GBA vs Pampeana")
plt.ylabel("Ingreso mediana real (deflactado)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"ingreso_mediana_real.png"))
plt.close()

print("Gráficos guardados en:", FIG_DIR)

# ---------------- 10) Imputación de ingresos faltantes (RandomForest) ----------------
# Preparamos dataset con features sencillas
print("\nImputación de ingresos faltantes con RandomForest...")
cols_for_model = []
if 'sexo' in eph.columns: cols_for_model.append('sexo')
if 'edad' in eph.columns: cols_for_model.append('edad')
if 'CONDICION' in eph.columns: cols_for_model.append('CONDICION')
if 'REGION_NAME' in eph.columns: cols_for_model.append('REGION_NAME')
# one-hot encode categóricas
df_model = eph[cols_for_model + ['ingreso_real','PONDIIO']].copy()
df_model = pd.get_dummies(df_model, columns=[c for c in cols_for_model if eph[c].dtype==object], drop_first=True)
# separar filas con y sin ingreso_real
train_df = df_model[~df_model['ingreso_real'].isna()].drop(columns=['PONDIIO'])
test_df = df_model[df_model['ingreso_real'].isna()].drop(columns=['PONDIIO'])
if len(train_df) < 100:
    print("Advertencia: pocas filas con ingreso observado. Imputación puede ser pobre.")
else:
    X = train_df.drop(columns=['ingreso_real'])
    y = train_df['ingreso_real']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("RF R2:", r2_score(y_test,y_pred), "RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
    # imputar en dataset original
    if len(test_df)>0:
        X_missing = test_df.drop(columns=['ingreso_real'])
        preds = rf.predict(X_missing)
        # asignar preds de vuelta a eph (matching por index)
        missing_idx = df_model[df_model['ingreso_real'].isna()].index
        eph.loc[missing_idx, 'ingreso_real'] = preds
        print("Imputadas", len(preds), "filas con ingreso_real.")
    # guardar modelo importances
    try:
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        importances.head(30).to_csv(os.path.join(OUTPUT_DIR,"rf_feature_importances.csv"))
    except Exception:
        pass

# actualizar indicadores con ingresos imputados (opcional)
ind2 = indicadores_por_periodo(eph)
ind2.to_csv(os.path.join(OUTPUT_DIR,"indicadores_univariado_ponderados_imputados.csv"), index=False)
print("Indicadores (con imputación) guardados.")

print("\nPIPELINE COMPLETADO. Revisá carpeta:", OUTPUT_DIR, "y figuras en:", FIG_DIR)
