import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

print("🚀 INICIANDO SIMULADOR FINANCIERO ATP CON ARCHIVOS LOCALES EXCEL...\n")

# --- 1. CARGA DE DATOS LOCALES (.xlsx) ---
print("📥 Leyendo archivos locales (2023.xlsx y 2024.xlsx)...")
try:
    df_train_raw = pd.read_excel("data/tenis/tenis_2023.xlsx")
    df_test_raw = pd.read_excel("data/tenis/tenis_2024.xlsx")
except FileNotFoundError:
    print("❌ Error: No encuentro los archivos. Asegúrate de que se llaman '2023.xlsx' y '2024.xlsx' y están en la misma carpeta que este script.")
    exit()
except ImportError:
    print("❌ Error: Falta una librería. Ejecuta en tu terminal: pip install openpyxl")
    exit()

# --- 2. INGENIERÍA DE CARACTERÍSTICAS Y LA "MONEDA AL AIRE" ---
def preparar_datos_apuestas(df):
    # Limpiamos filas sin cuotas de Bet365 o sin Ranking
    df = df.dropna(subset=['WRank', 'LRank', 'B365W', 'B365L', 'Surface']).copy()
    
    # Aseguramos que los rankings sean números
    df['WRank'] = pd.to_numeric(df['WRank'], errors='coerce')
    df['LRank'] = pd.to_numeric(df['LRank'], errors='coerce')
    df = df.dropna(subset=['WRank', 'LRank']).copy()
    
    moneda = np.random.randint(0, 2, size=len(df))
    df_limpio = pd.DataFrame()
    
    # Diferencia de Ranking
    rank_A = np.where(moneda == 1, df['WRank'], df['LRank'])
    rank_B = np.where(moneda == 1, df['LRank'], df['WRank'])
    df_limpio['Dif_Ranking'] = rank_A - rank_B
    
    # Superficie (One-Hot Encoding)
    df_limpio['Surface'] = df['Surface'].values
    df_limpio = pd.get_dummies(df_limpio, columns=['Surface'], drop_first=False)
    
    # Asignamos las cuotas al Jugador A y al Jugador B según la moneda
    df_limpio['Cuota_A'] = np.where(moneda == 1, df['B365W'], df['B365L'])
    df_limpio['Cuota_B'] = np.where(moneda == 1, df['B365L'], df['B365W'])
    
    df_limpio['Target'] = moneda
    return df_limpio

print("⚙️ Procesando datos y anonimizando jugadores...")
df_train = preparar_datos_apuestas(df_train_raw)
df_test = preparar_datos_apuestas(df_test_raw)

# Alineamos columnas por si hay superficies diferentes entre un año y otro
X_train_full, X_test_full = df_train.align(df_test, join='inner', axis=1)

# Separamos las cuotas del set de entrenamiento para que la IA no haga trampas
X_train = X_train_full.drop(['Target', 'Cuota_A', 'Cuota_B'], axis=1)
y_train = df_train['Target']

X_test = X_test_full.drop(['Target', 'Cuota_A', 'Cuota_B'], axis=1)
y_test = df_test['Target']

# --- 3. ENTRENAMIENTO DEL MODELO CALIBRADO ---
print("🧠 Entrenando al modelo predictivo base (XGBoost)...")
xgb_base = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)

print("⚖️ Aplicando Calibración Matemática (Platt Scaling) para ajustar probabilidades...")
# Envolvemos nuestro XGBoost en el calibrador (cv=5 divide los datos en 5 bloques para aprender sus errores)
modelo = CalibratedClassifierCV(estimator=xgb_base, method='sigmoid', cv=5)
modelo.fit(X_train, y_train)

# --- 4. SIMULADOR DE INVERSIÓN (CRITERIO DE KELLY) ---
capital_inicial = 1000.0
capital_actual = capital_inicial
apuestas_realizadas = 0
apuestas_ganadas = 0

probabilidades = modelo.predict_proba(X_test)
predicciones = modelo.predict(X_test)

print("💸 Iniciando simulación de apuestas sobre la temporada 2024...\n")

for i in range(len(X_test)):
    prediccion_ia = predicciones[i]
    resultado_real = y_test.iloc[i]
    
    # Determinamos la cuota y la probabilidad según lo que elija la IA
    if prediccion_ia == 1:
        cuota_jugada = df_test.iloc[i]['Cuota_A']
        prob_ia = probabilidades[i][1]
    else:
        cuota_jugada = df_test.iloc[i]['Cuota_B']
        prob_ia = probabilidades[i][0]
        
    prob_casa = 1 / cuota_jugada
    
    # SOLO APOSTAMOS SI HAY VALOR
    if prob_ia > prob_casa:
        p = prob_ia
        q = 1 - p
        b = cuota_jugada - 1.0 
        
        fraccion_kelly = (p * b - q) / b
        # Kelly Fraccional (25%) para mitigar el riesgo
        kelly_seguro = fraccion_kelly * 0.25
        
        if kelly_seguro > 0.05:
            kelly_seguro = 0.05
            
        if kelly_seguro > 0:
            cantidad_apostada = capital_actual * kelly_seguro
            capital_actual -= cantidad_apostada
            apuestas_realizadas += 1
            
            if prediccion_ia == resultado_real:
                capital_actual += (cantidad_apostada * cuota_jugada)
                apuestas_ganadas += 1

# --- 5. RESULTADOS FINALES ---
roi = ((capital_actual - capital_inicial) / capital_inicial) * 100

print("-" * 40)
print(f"📊 Partidos totales en la base de datos: {len(X_test)}")
print(f"🎯 Apuestas realizadas (Value Bets): {apuestas_realizadas}")
print(f"✅ Apuestas ganadas: {apuestas_ganadas}")
if apuestas_realizadas > 0:
    print(f"📈 Porcentaje de acierto real apostando: {(apuestas_ganadas/apuestas_realizadas)*100:.1f}%")
print(f"💰 CAPITAL INICIAL: {capital_inicial:.2f}€")
print(f"💸 CAPITAL FINAL:   {capital_actual:.2f}€")
print(f"🚀 R.O.I. TOTAL:    {roi:.2f}%")
print("-" * 40)