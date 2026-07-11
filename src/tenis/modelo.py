import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print("🚀 INICIANDO MEGA-MODELO ATP (RANKING + EFECTIVIDAD POR SUPERFICIE)...\n")

# 1. CARGA DE DATOS (2019-2024)
años = [2019, 2020, 2021, 2022, 2023]
print("Descargando años de entrenamiento...")
df_train_raw = pd.concat([pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{a}.csv") for a in años], ignore_index=True)
df_test_raw = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv")

df_total = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
df_total['match_id'] = range(len(df_total))

# --- 2. MAGIA DE PANDAS AVANZADA: HISTÓRICO DE VICTORIAS ---
print("📊 Calculando % de victorias históricas (General y por Superficie)...")

# Creamos una tabla con TODAS las actuaciones individuales
ganadores = df_total[['match_id', 'tourney_date', 'winner_name', 'surface']].copy()
ganadores.columns = ['match_id', 'date', 'player', 'surface']
ganadores['won'] = 1 # 1 = Victoria

perdedores = df_total[['match_id', 'tourney_date', 'loser_name', 'surface']].copy()
perdedores.columns = ['match_id', 'date', 'player', 'surface']
perdedores['won'] = 0 # 0 = Derrota

actuaciones = pd.concat([ganadores, perdedores]).sort_values(['player', 'date'])

# Calculamos % de victorias GENERAL acumulado (shift(1) para no ver el futuro)
actuaciones['win_pct_general'] = actuaciones.groupby('player')['won'].transform(lambda x: x.shift(1).expanding().mean())

# Calculamos % de victorias EN ESA SUPERFICIE acumulado
actuaciones['win_pct_surface'] = actuaciones.groupby(['player', 'surface'])['won'].transform(lambda x: x.shift(1).expanding().mean())

# Rellenamos los NaN (partidos debut) con 0.5 (50%)
actuaciones.fillna({'win_pct_general': 0.5, 'win_pct_surface': 0.5}, inplace=True)

# Mapeamos estos datos de vuelta al DataFrame original
dict_general = actuaciones.set_index(['match_id', 'player'])['win_pct_general'].to_dict()
dict_surface = actuaciones.set_index(['match_id', 'player'])['win_pct_surface'].to_dict()

df_total['w_win_pct_gen'] = df_total.set_index(['match_id', 'winner_name']).index.map(dict_general)
df_total['l_win_pct_gen'] = df_total.set_index(['match_id', 'loser_name']).index.map(dict_general)
df_total['w_win_pct_sur'] = df_total.set_index(['match_id', 'winner_name']).index.map(dict_surface)
df_total['l_win_pct_sur'] = df_total.set_index(['match_id', 'loser_name']).index.map(dict_surface)

# Volvemos a separar Train y Test
df_train_raw = df_total[df_total['tourney_date'] < 20240000].copy()
df_test_raw = df_total[df_total['tourney_date'] > 20240000].copy()

# --- 3. APLICAMOS LA MONEDA Y CREAMOS LAS DIFERENCIAS ---
def preparar_datos_final(df):
    df = df.dropna(subset=['winner_rank', 'loser_rank', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age']).copy()
    moneda = np.random.randint(0, 2, size=len(df))
    df_limpio = pd.DataFrame()
    
    # ¡VUELVE EL RANKING!
    rank_A = np.where(moneda == 1, df['winner_rank'], df['loser_rank'])
    rank_B = np.where(moneda == 1, df['loser_rank'], df['winner_rank'])
    df_limpio['Dif_Ranking'] = rank_A - rank_B
    
    # Diferencia Físicas
    df_limpio['Dif_Edad'] = np.where(moneda == 1, df['winner_age'], df['loser_age']) - np.where(moneda == 1, df['loser_age'], df['winner_age'])
    df_limpio['Dif_Altura'] = np.where(moneda == 1, df['winner_ht'], df['loser_ht']) - np.where(moneda == 1, df['loser_ht'], df['winner_ht'])
    
    # NUEVAS SUPER-VARIABLES: Diferencia de % Victorias
    pct_gen_A = np.where(moneda == 1, df['w_win_pct_gen'], df['l_win_pct_gen'])
    pct_gen_B = np.where(moneda == 1, df['l_win_pct_gen'], df['w_win_pct_gen'])
    df_limpio['Dif_WinPct_General'] = pct_gen_A - pct_gen_B
    
    pct_sur_A = np.where(moneda == 1, df['w_win_pct_sur'], df['l_win_pct_sur'])
    pct_sur_B = np.where(moneda == 1, df['l_win_pct_sur'], df['w_win_pct_sur'])
    df_limpio['Dif_WinPct_Superficie'] = pct_sur_A - pct_sur_B
    
    df_limpio['Target'] = moneda
    return df_limpio

df_train = preparar_datos_final(df_train_raw)
df_test = preparar_datos_final(df_test_raw)

X_train = df_train.drop('Target', axis=1)
y_train = df_train['Target']
X_test = df_test.drop('Target', axis=1)
y_test = df_test['Target']

# --- 4. ENTRENAMIENTO Y EVALUACIÓN ---
print("🧠 Entrenando XGBoost con datos históricos de victorias...")
modelo_tenis = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)
modelo_tenis.fit(X_train, y_train)

predicciones = modelo_tenis.predict(X_test)
precision = accuracy_score(y_test, predicciones)

print("\n" + "="*40)
print(f"🎯 Precisión Final 2024: {precision * 100:.2f}%")
print("="*40)

# --- 5. IMPORTANCIA DE VARIABLES ---
print("\n🧠 IMPORTANCIA DE LAS VARIABLES (¿A qué hace caso el modelo?):")
importancias = pd.DataFrame({
    'Variable': X_train.columns,
    'Importancia': modelo_tenis.feature_importances_ * 100
}).sort_values(by='Importancia', ascending=False)
print(importancias.to_string(index=False))