import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore') # Para que Pandas no ensucie la terminal con avisos

print("🚀 INICIANDO EL BOT CUANTITATIVO DEFINITIVO DE TENIS 🚀\n")

# --- 1. CARGA DE DATOS HISTÓRICOS (SACKMANN 2019-2024) ---
print("📥 1/5 Descargando la historia del tenis (2019-2024)...")
años = [2019, 2020, 2021, 2022, 2023, 2024]
df_total = pd.concat([pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{a}.csv") for a in años], ignore_index=True)
df_total['match_id'] = range(len(df_total))

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (MEMORIA DEL JUGADOR) ---
print("🧠 2/5 Calculando el estado de forma histórico (% Saques y Victorias)...")
actuaciones = pd.concat([
    df_total[['match_id', 'tourney_date', 'winner_name', 'w_1stIn', 'w_svpt', 'surface']].rename(columns={'winner_name':'player', 'w_1stIn':'1stIn', 'w_svpt':'svpt'}),
    df_total[['match_id', 'tourney_date', 'loser_name', 'l_1stIn', 'l_svpt', 'surface']].rename(columns={'loser_name':'player', 'l_1stIn':'1stIn', 'l_svpt':'svpt'})
]).sort_values(['player', 'tourney_date'])

actuaciones['won'] = np.concatenate([np.ones(len(df_total)), np.zeros(len(df_total))])
actuaciones['pct_1stIn'] = actuaciones['1stIn'] / actuaciones['svpt']

# Medias móviles (shift(1) para no hacer trampas)
actuaciones['hist_1stIn'] = actuaciones.groupby('player')['pct_1stIn'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
actuaciones['win_pct_gen'] = actuaciones.groupby('player')['won'].transform(lambda x: x.shift(1).expanding().mean())

actuaciones.fillna({'hist_1stIn': 0.60, 'win_pct_gen': 0.50}, inplace=True) # Valores neutros para debutantes

dict_saque = actuaciones.set_index(['match_id', 'player'])['hist_1stIn'].to_dict()
dict_win = actuaciones.set_index(['match_id', 'player'])['win_pct_gen'].to_dict()

df_total['w_hist_1stIn'] = df_total.set_index(['match_id', 'winner_name']).index.map(dict_saque)
df_total['l_hist_1stIn'] = df_total.set_index(['match_id', 'loser_name']).index.map(dict_saque)
df_total['w_win_pct_gen'] = df_total.set_index(['match_id', 'winner_name']).index.map(dict_win)
df_total['l_win_pct_gen'] = df_total.set_index(['match_id', 'loser_name']).index.map(dict_win)

# --- 3. SEPARACIÓN Y FUSIÓN DE CUOTAS PARA 2024 ---
print("🔗 3/5 Fusionando las estadísticas de 2024 con el Excel de Bet365...")
df_train_raw = df_total[df_total['tourney_date'] < 20240000].copy()
df_test_stats = df_total[df_total['tourney_date'] > 20240000].copy()

try:
    df_odds = pd.read_excel("data/tenis/tenis_2024.xlsx")
except FileNotFoundError:
    print("❌ Error: No encuentro el archivo '2024.xlsx'.")
    exit()

# Algoritmo de Fusión
df_test_stats['w_apellido'] = df_test_stats['winner_name'].astype(str).apply(lambda x: x.split()[-1].lower())
df_test_stats['l_apellido'] = df_test_stats['loser_name'].astype(str).apply(lambda x: x.split()[-1].lower())
df_test_stats['mes'] = (df_test_stats['tourney_date'] // 100) % 100

df_odds['w_apellido'] = df_odds['Winner'].astype(str).apply(lambda x: x.split()[0].lower())
df_odds['l_apellido'] = df_odds['Loser'].astype(str).apply(lambda x: x.split()[0].lower())
df_odds['mes'] = pd.to_datetime(df_odds['Date']).dt.month

df_test_raw = pd.merge(df_test_stats, df_odds[['w_apellido', 'l_apellido', 'mes', 'B365W', 'B365L']], 
                       on=['w_apellido', 'l_apellido', 'mes'], how='inner').drop_duplicates(subset=['tourney_name', 'w_apellido', 'l_apellido'])

# --- 4. LA MONEDA Y EL MODELO CALIBRADO ---
def preparar_datos_finales(df, is_test=False):
    columnas_req = ['winner_rank', 'loser_rank', 'w_hist_1stIn', 'l_hist_1stIn', 'w_win_pct_gen', 'l_win_pct_gen', 'winner_age', 'loser_age']
    if is_test: columnas_req += ['B365W', 'B365L']
    
    df = df.dropna(subset=columnas_req).copy()
    moneda = np.random.randint(0, 2, size=len(df))
    df_limpio = pd.DataFrame()
    
    # Variables de Entrenamiento
    df_limpio['Dif_Ranking'] = np.where(moneda == 1, df['winner_rank'], df['loser_rank']) - np.where(moneda == 1, df['loser_rank'], df['winner_rank'])
    df_limpio['Dif_Saque'] = np.where(moneda == 1, df['w_hist_1stIn'], df['l_hist_1stIn']) - np.where(moneda == 1, df['l_hist_1stIn'], df['w_hist_1stIn'])
    df_limpio['Dif_WinPct'] = np.where(moneda == 1, df['w_win_pct_gen'], df['l_win_pct_gen']) - np.where(moneda == 1, df['l_win_pct_gen'], df['w_win_pct_gen'])
    df_limpio['Dif_Edad'] = np.where(moneda == 1, df['winner_age'], df['loser_age']) - np.where(moneda == 1, df['loser_age'], df['winner_age'])
    
    if is_test:
        df_limpio['Cuota_A'] = np.where(moneda == 1, df['B365W'], df['B365L'])
        df_limpio['Cuota_B'] = np.where(moneda == 1, df['B365L'], df['B365W'])
        
    df_limpio['Target'] = moneda
    return df_limpio

df_train = preparar_datos_finales(df_train_raw)
df_test = preparar_datos_finales(df_test_raw, is_test=True)

X_train = df_train.drop('Target', axis=1)
y_train = df_train['Target']

X_test = df_test.drop(['Target', 'Cuota_A', 'Cuota_B'], axis=1)
y_test = df_test['Target']

print("⚖️ 4/5 Entrenando XGBoost y aplicando Calibración Matemática...")
xgb_base = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
modelo_calibrado = CalibratedClassifierCV(estimator=xgb_base, method='sigmoid', cv=5)
modelo_calibrado.fit(X_train, y_train)

# --- 5. SIMULADOR FINANCIERO Y ANÁLISIS DE TRAMOS ---
print("💸 5/5 Ejecutando el Bankroll y Analizando Tramos de Cuotas...")
STAKE_PLANO = 10.0
registro_apuestas = []

probabilidades = modelo_calibrado.predict_proba(X_test)
predicciones = modelo_calibrado.predict(X_test)

for i in range(len(X_test)):
    pred = predicciones[i]
    real = y_test.iloc[i]
    
    cuota = df_test.iloc[i]['Cuota_A'] if pred == 1 else df_test.iloc[i]['Cuota_B']
    prob_ia = probabilidades[i][1] if pred == 1 else probabilidades[i][0]
    prob_casa = 1 / cuota
    
    if prob_ia > (prob_casa + 0.05) and prob_ia >= 0.35 and cuota <= 3.00:
        acierto = 1 if pred == real else 0
        beneficio = (STAKE_PLANO * cuota) - STAKE_PLANO if acierto == 1 else -STAKE_PLANO
        
        registro_apuestas.append({
            'Cuota': cuota,
            'Acierto': acierto,
            'Beneficio': beneficio
        })

# --- 6. AUTOPSIA DE LOS RESULTADOS ---
df_resultados = pd.DataFrame(registro_apuestas)

print("\n" + "=" * 50)
print("🔍 AUTOPSIA DEL MODELO: RENTABILIDAD POR TRAMOS 🔍")
print("=" * 50)

if len(df_resultados) > 0:
    # Creamos tramos de cuotas para ver dónde somos buenos y dónde somos malos
    tramos = [1.0, 1.5, 2.0, 2.5, 3.0]
    etiquetas = ['Favoritos (1.0-1.5)', 'Igualados (1.5-2.0)', 'No Favoritos (2.0-2.5)', 'Sorpresas (2.5-3.0)']
    df_resultados['Tramo'] = pd.cut(df_resultados['Cuota'], bins=tramos, labels=etiquetas)
    
    # Calculamos las estadísticas por tramo
    resumen = df_resultados.groupby('Tramo').agg(
        Apuestas=('Beneficio', 'count'),
        Aciertos=('Acierto', 'sum'),
        Beneficio_Neto=('Beneficio', 'sum')
    )
    
    resumen['% Acierto'] = (resumen['Aciertos'] / resumen['Apuestas'] * 100).fillna(0).round(1)
    resumen['Yield %'] = ((resumen['Beneficio_Neto'] / (resumen['Apuestas'] * STAKE_PLANO)) * 100).fillna(0).round(2)
    
    print(resumen[['Apuestas', '% Acierto', 'Beneficio_Neto', 'Yield %']])
    
    beneficio_total = df_resultados['Beneficio'].sum()
    yield_total = (beneficio_total / (len(df_resultados) * STAKE_PLANO)) * 100
    print("-" * 50)
    print(f"💰 BENEFICIO TOTAL: {beneficio_total:+.2f}€")
    print(f"🚀 YIELD GLOBAL:    {yield_total:+.2f}%")
else:
    print("No se registraron apuestas con los filtros actuales.")
print("=" * 50)