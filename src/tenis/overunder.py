import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# --- 1. CARGA DE DATOS (igual que antes) ---
años = [2019, 2020, 2021, 2022, 2023, 2024]
df_total = pd.concat([
    pd.read_csv(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{a}.csv")
    for a in años], ignore_index=True)
df_total = df_total[~df_total['score'].str.contains('RET|W/O|DEF', na=False)].copy()
df_total = df_total[df_total['best_of'] == 3].copy()

# --- 2. ANALIZAR MARCADOR (igual que antes) ---
def analizar_marcador(marcador):
    try:
        if pd.isna(marcador): return np.nan, np.nan
        sets = marcador.split()
        total_juegos, tie_breaks = 0, 0
        for s in sets:
            if '(' in s: tie_breaks += 1
            juegos = s.split('(')[0].split('-')
            if len(juegos) == 2:
                total_juegos += int(juegos[0]) + int(juegos[1])
        return total_juegos, tie_breaks
    except:
        return np.nan, np.nan

res = df_total['score'].apply(analizar_marcador)
df_total['Total_Juegos'] = [r[0] for r in res]
df_total['Tie_Breaks_Partido'] = [r[1] for r in res]
df_total = df_total.dropna(subset=['Total_Juegos'])
df_total['match_id'] = range(len(df_total))

# --- 3. HISTORIAL DE JUGADORES: AÑADIMOS RETURN POINTS ---
# CORRECCIÓN CLAVE: win_pct en el servicio del rival (mide capacidad de break)
ganadores = df_total[['match_id', 'tourney_date', 'winner_name',
                       'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                       'l_svpt', 'l_1stWon', 'l_2ndWon',  # puntos de resto del ganador
                       'Tie_Breaks_Partido']].copy()

# Puntos de resto ganados por el ganador (break potential)
# l_svpt - l_1stWon - l_2ndWon = puntos que el ganador ganó devolviendo
ganadores['return_pts_won'] = (
    ganadores['l_svpt'] - ganadores['l_1stWon'] - ganadores['l_2ndWon']
)
ganadores['return_pct'] = ganadores['return_pts_won'] / ganadores['l_svpt'].clip(lower=1)
ganadores['ace_pct'] = ganadores['w_ace'] / ganadores['w_svpt'].clip(lower=1)
ganadores['df_pct'] = ganadores['w_df'] / ganadores['w_svpt'].clip(lower=1)
ganadores['tb_pct'] = ganadores['Tie_Breaks_Partido'] / 2  # aprox sets jugados
ganadores = ganadores[['match_id', 'tourney_date', 'winner_name',
                        'ace_pct', 'df_pct', 'return_pct', 'tb_pct']].copy()
ganadores.columns = ['match_id', 'date', 'player', 'ace_pct', 'df_pct', 'return_pct', 'tb_pct']

perdedores = df_total[['match_id', 'tourney_date', 'loser_name',
                        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
                        'w_svpt', 'w_1stWon', 'w_2ndWon',
                        'Tie_Breaks_Partido']].copy()
perdedores['return_pts_won'] = (
    perdedores['w_svpt'] - perdedores['w_1stWon'] - perdedores['w_2ndWon']
)
perdedores['return_pct'] = perdedores['return_pts_won'] / perdedores['w_svpt'].clip(lower=1)
perdedores['ace_pct'] = perdedores['l_ace'] / perdedores['l_svpt'].clip(lower=1)
perdedores['df_pct'] = perdedores['l_df'] / perdedores['l_svpt'].clip(lower=1)
perdedores['tb_pct'] = perdedores['Tie_Breaks_Partido'] / 2
perdedores = perdedores[['match_id', 'tourney_date', 'loser_name',
                          'ace_pct', 'df_pct', 'return_pct', 'tb_pct']].copy()
perdedores.columns = ['match_id', 'date', 'player', 'ace_pct', 'df_pct', 'return_pct', 'tb_pct']

actuaciones = pd.concat([ganadores, perdedores]).sort_values(['player', 'date'])

for col in ['ace_pct', 'df_pct', 'return_pct', 'tb_pct']:
    actuaciones[f'hist_{col}'] = actuaciones.groupby('player')[col].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

defaults = {'hist_ace_pct': 0.06, 'hist_df_pct': 0.03,
            'hist_return_pct': 0.37, 'hist_tb_pct': 0.15}
actuaciones.fillna(defaults, inplace=True)

# Construir diccionarios de lookup
dicts = {}
for col in ['hist_ace_pct', 'hist_df_pct', 'hist_return_pct', 'hist_tb_pct']:
    dicts[col] = actuaciones.set_index(['match_id', 'player'])[col].to_dict()

for col in dicts:
    df_total[f'w_{col}'] = df_total.set_index(['match_id', 'winner_name']).index.map(dicts[col])
    df_total[f'l_{col}'] = df_total.set_index(['match_id', 'loser_name']).index.map(dicts[col])

# --- 4. PREPARACIÓN: CORRECCIÓN PRINCIPAL ---
def preparar_features(df):
    df = df.dropna(subset=[
        'winner_rank', 'loser_rank', 'Total_Juegos',
        'w_hist_ace_pct', 'l_hist_ace_pct',
        'w_hist_return_pct', 'l_hist_return_pct'
    ]).copy()

    moneda = np.random.randint(0, 2, size=len(df))
    out = pd.DataFrame()

    # Features de cada jugador (A y B aleatorios)
    for stat in ['hist_ace_pct', 'hist_df_pct', 'hist_return_pct', 'hist_tb_pct']:
        A = np.where(moneda == 1, df[f'w_{stat}'], df[f'l_{stat}'])
        B = np.where(moneda == 1, df[f'l_{stat}'], df[f'w_{stat}'])

        out[f'A_{stat}'] = A
        out[f'B_{stat}'] = B

        # CORRECCIÓN 2: Features de interacción, no solo sumas
        out[f'sum_{stat}'] = A + B          # potencia conjunta
        out[f'diff_{stat}'] = abs(A - B)    # asimetría entre jugadores

    # CORRECCIÓN CLAVE: saque de A vs devolución de B (y viceversa)
    # Esto predice cuántos breaks habrá
    out['ace_A_vs_return_B'] = out['A_hist_ace_pct'] * (1 - out['B_hist_return_pct'])
    out['ace_B_vs_return_A'] = out['B_hist_ace_pct'] * (1 - out['A_hist_return_pct'])
    out['break_potential'] = out['B_hist_return_pct'] + out['A_hist_return_pct']

    rank_A = np.where(moneda == 1, df['winner_rank'], df['loser_rank'])
    rank_B = np.where(moneda == 1, df['loser_rank'], df['winner_rank'])
    out['desequilibrio_ranking'] = abs(rank_A - rank_B)
    out['log_ranking_ratio'] = np.log1p(out['desequilibrio_ranking'])

    # Superficie
    out['surface'] = df['surface'].values
    out = pd.get_dummies(out, columns=['surface'], drop_first=False)

    # TARGET 1: número exacto de juegos (regresión)
    out['target_juegos'] = df['Total_Juegos'].values

    return out

# --- 5. ENTRENAMIENTO EN DOS FASES ---
df_train_raw = df_total[df_total['tourney_date'] < 20240000]
df_test_raw  = df_total[df_total['tourney_date'] >= 20240000]

df_train = preparar_features(df_train_raw)
df_test  = preparar_features(df_test_raw)

feature_cols = [c for c in df_train.columns if c != 'target_juegos']
X_train, X_test = df_train[feature_cols].align(df_test[feature_cols], join='left', axis=1, fill_value=0)
y_train = df_train['target_juegos']
y_test  = df_test['target_juegos']

print("FASE 1 — Regresión: predecir total de juegos exacto")
modelo_reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
modelo_reg.fit(X_train, y_train,
               eval_set=[(X_test, y_test)],
               verbose=False)

pred_juegos = modelo_reg.predict(X_test)
mae = mean_absolute_error(y_test, pred_juegos)
print(f"MAE en 2024: {mae:.2f} juegos")

# Distribución de predicciones
print(f"Media predicha: {pred_juegos.mean():.1f} | Real: {y_test.mean():.1f}")
print(f"Std predicha:   {pred_juegos.std():.1f} | Real: {y_test.std():.1f}")

# --- 6. FASE 2: SIMULADOR FINANCIERO AVANZADO (LÍNEA DINÁMICA POR PROBABILIDAD) ---
print("\n🔗 FASE 2 — Simulador Financiero con Líneas Dinámicas Proxy")

# 1. Cargamos tu archivo de cuotas para el set de Test (2024)
try:
    df_odds = pd.read_excel("tenis_2024.xlsx")
    df_odds['w_apellido'] = df_odds['Winner'].astype(str).apply(lambda x: x.split()[0].lower())
    df_odds['l_apellido'] = df_odds['Loser'].astype(str).apply(lambda x: x.split()[0].lower())
    df_odds['mes'] = pd.to_datetime(df_odds['Date']).dt.month
    
    # Preparamos los nombres en nuestro Test Set para poder cruzarlos
    df_test_raw['w_apellido'] = df_test_raw['winner_name'].astype(str).apply(lambda x: x.split()[-1].lower())
    df_test_raw['l_apellido'] = df_test_raw['loser_name'].astype(str).apply(lambda x: x.split()[-1].lower())
    df_test_raw['mes'] = (df_test_raw['tourney_date'] // 100) % 100
    
    # Fusionamos para traernos las cuotas B365W y B365L
    df_cruce = pd.merge(df_test_raw, df_odds[['w_apellido', 'l_apellido', 'mes', 'B365W', 'B365L']], 
                        on=['w_apellido', 'l_apellido', 'mes'], how='inner').drop_duplicates(subset=['match_id'])
    
    # Alineamos nuestras predicciones con el cruce exitoso
    indices_validos = df_cruce.index.intersection(X_test.index)
    X_test_cruce = X_test.loc[indices_validos]
    y_test_cruce = y_test.loc[indices_validos]
    pred_juegos_cruce = modelo_reg.predict(X_test_cruce)
    
except FileNotFoundError:
    print("❌ Error: Necesitas el archivo '2024.xlsx' en la carpeta para simular las líneas.")
    exit()

# 2. Calculamos la Probabilidad Implícita del Favorito (Quitando el Vig/Comisión)
df_cruce['p_raw_W'] = 1 / df_cruce['B365W']
df_cruce['p_raw_L'] = 1 / df_cruce['B365L']
df_cruce['vig'] = df_cruce['p_raw_W'] + df_cruce['p_raw_L']
# Probabilidad de victoria de ambos sin el margen de la casa
df_cruce['p_W_limpia'] = df_cruce['p_raw_W'] / df_cruce['vig']
df_cruce['p_L_limpia'] = df_cruce['p_raw_L'] / df_cruce['vig']

# Extraemos la probabilidad del jugador que es FAVORITO (el que tiene más de 50%)
df_cruce['p_favorito'] = df_cruce[['p_W_limpia', 'p_L_limpia']].max(axis=1)

# 3. Función del Espejo Estadístico (Línea Dinámica)
def estimar_linea_casa(p_fav, superficie):
    """
    Estima la línea de O/U de Bet365 basándose en cómo de favorito es el jugador.
    Si la probabilidad del favorito es gigante (ej. 90%), la línea baja a 19.5 o 20.5.
    Si está muy igualado (ej. 55%), la línea sube a 22.5 o 23.5.
    """
    # Ajuste base por superficie (Hierba siempre suma medio juego, Tierra resta medio)
    ajuste_sup = 0.5 if superficie == 'Grass' else (-0.5 if superficie == 'Clay' else 0)
    
    if p_fav >= 0.85: return 20.5 + ajuste_sup # Cuota ~1.10
    elif p_fav >= 0.75: return 21.5 + ajuste_sup # Cuota ~1.25
    elif p_fav >= 0.60: return 22.5 + ajuste_sup # Cuota ~1.50
    else: return 23.5 + ajuste_sup # Partido muy igualado (Cuota > 1.65)

# 4. El Simulador de Inversión
resultados_sim = []
CUOTA_OU = 1.85 # Cuota estándar del mercado Over/Under en Bet365
MARGEN_CONVICCION = 2.0 # Exigimos a nuestra IA discrepar al menos 2 juegos de la casa

for i, (idx, row) in enumerate(df_cruce.iterrows()):
    pred = pred_juegos_cruce[i]
    real = y_test_cruce.iloc[i]
    p_fav = row['p_favorito']
    sup = row['surface']
    
    linea_casa = estimar_linea_casa(p_fav, sup)
    
    if pred > linea_casa + MARGEN_CONVICCION:
        apuesta = 'OVER'
        gana = 1 if real > linea_casa else 0
    elif pred < linea_casa - MARGEN_CONVICCION:
        apuesta = 'UNDER'
        gana = 1 if real <= linea_casa else 0
    else:
        apuesta = 'SKIP'
        gana = None
        
    resultados_sim.append({
        'Prob_Fav': round(p_fav*100, 1), 'Linea_Casa': linea_casa, 
        'Pred_IA': round(pred, 1), 'Juegos_Reales': real,
        'Apuesta': apuesta, 'Gana': gana
    })

# 5. Evaluación Financiera Final
df_sim = pd.DataFrame(resultados_sim)
apostadas = df_sim[df_sim['Apuesta'] != 'SKIP']
skipped = df_sim[df_sim['Apuesta'] == 'SKIP']

print(f"Partidos analizados con cuotas: {len(df_sim)}")
print(f"Apuestas realizadas:            {len(apostadas)} ({len(apostadas)/len(df_sim)*100:.1f}%)")

if len(apostadas) > 0:
    acierto = apostadas['Gana'].mean()
    # Calcular el Yield real asumiendo que apostamos 10€ a cuota 1.85
    beneficio_por_apuesta_ganada = (CUOTA_OU - 1)
    beneficio_por_apuesta_perdida = -1
    
    ganancias = apostadas['Gana'] * beneficio_por_apuesta_ganada + (1 - apostadas['Gana']) * beneficio_por_apuesta_perdida
    yield_pct = ganancias.mean() * 100
    
    print(f"🎯 Accuracy (Acierto):          {acierto*100:.1f}%")
    print(f"⚠️ Punto de Equilibrio (BE):    54.1% (Para cuota 1.85)")
    print("-" * 50)
    print(f"🚀 YIELD NETO ESTIMADO:         {yield_pct:+.2f}%")
    print("-" * 50)
    
    print("\nDesglose por tipo:")
    for tipo in ['OVER', 'UNDER']:
        sub = apostadas[apostadas['Apuesta'] == tipo]
        if len(sub) > 0:
            print(f"  {tipo}: {len(sub)} apuestas | {sub['Gana'].mean()*100:.1f}% acierto")
else:
    print("El margen de convicción (2.0) es tan estricto que no se han encontrado apuestas.")