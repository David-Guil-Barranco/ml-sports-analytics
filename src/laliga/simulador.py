import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("🚀 INICIANDO MEGA-SIMULADOR A LARGO PLAZO (5 TEMPORADAS) 🚀\n")

# --- 1. CARGA Y FUSIÓN DE DATOS ---
# Cambia estos nombres por los de tus archivos descargados, ordenados del más antiguo al más nuevo
archivos_csv = [
    'data/laliga/SP1_21-22.csv',
    'data/laliga/SP1_22-23.csv',
    'data/laliga/SP1_23-24.csv',
    'data/laliga/SP1_24-25.csv',
    'data/laliga/SP1_25-26.csv'
]

lista_dataframes = []
for archivo in archivos_csv:
    try:
        df_temp = pd.read_csv(archivo)
        lista_dataframes.append(df_temp)
    except FileNotFoundError:
        print(f"⚠️ ¡Cuidado! No se encuentra: {archivo}")

# Unimos las 5 temporadas en una sola macro-tabla
df_laliga = pd.concat(lista_dataframes, ignore_index=True)

# Filtramos columnas (añadimos Goles, Tiros, Córners y Faltas)
columnas_clave = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']
df_limpio = df_laliga[columnas_clave].copy()
df_limpio['Target'] = df_limpio['FTR'].map({'D': 0, 'H': 1, 'A': 2})

# --- 2. MEGA INGENIERÍA DE CARACTERÍSTICAS ---
print("Calculando estadísticas históricas de casi 2000 partidos...")
df_limpio['Media_Tiros_Local'] = df_limpio.groupby('HomeTeam')['HST'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['Media_Tiros_Visitante'] = df_limpio.groupby('AwayTeam')['AST'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['GF_Local'] = df_limpio.groupby('HomeTeam')['FTHG'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['GC_Local'] = df_limpio.groupby('HomeTeam')['FTAG'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['GF_Visitante'] = df_limpio.groupby('AwayTeam')['FTAG'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['GC_Visitante'] = df_limpio.groupby('AwayTeam')['FTHG'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['Corners_Local'] = df_limpio.groupby('HomeTeam')['HC'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['Corners_Visitante'] = df_limpio.groupby('AwayTeam')['AC'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['Faltas_Local'] = df_limpio.groupby('HomeTeam')['HF'].shift(1).rolling(3, min_periods=1).mean()
df_limpio['Faltas_Visitante'] = df_limpio.groupby('AwayTeam')['AF'].shift(1).rolling(3, min_periods=1).mean()

df_modelo = df_limpio.dropna()

# --- 3. PREPARACIÓN Y ENTRENAMIENTO DEL MODELO ---
columnas_X = [
    'B365H', 'B365D', 'B365A', 
    'Media_Tiros_Local', 'Media_Tiros_Visitante',
    'GF_Local', 'GC_Local', 'GF_Visitante', 'GC_Visitante',
    'Corners_Local', 'Corners_Visitante',
    'Faltas_Local', 'Faltas_Visitante'
]

X = df_modelo[columnas_X]
y = df_modelo['Target']

# shuffle=False asegura que el test_size=0.2 coja EXCLUSIVAMENTE la última temporada para simular
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# XGBoost un poco más conservador (max_depth=2) para evitar sobreajustar tantos datos
modelo_xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)
modelo_xgb.fit(X_train, y_train)

# --- 4. SIMULADOR DE BANKROLL CON KELLY FRACCIONAL ---
capital_inicial = 1000.0
capital_actual = capital_inicial
apuestas_realizadas = 0
apuestas_ganadas = 0

probabilidades_modelo = modelo_xgb.predict_proba(X_test)
predicciones = modelo_xgb.predict(X_test)

print("Iniciando simulación de apuestas sobre la última temporada...\n")

for i in range(len(X_test)):
    fila = X_test.iloc[i]
    resultado_real = y_test.iloc[i]
    prediccion_ia = predicciones[i]
    
    if prediccion_ia == 1:
        cuota_jugada = fila['B365H']
    elif prediccion_ia == 0:
        cuota_jugada = fila['B365D']
    else:
        cuota_jugada = fila['B365A']
        
    prob_casa = 1 / cuota_jugada
    prob_ia = probabilidades_modelo[i][prediccion_ia]
    
    # Value Bet detectada
    if prob_ia > prob_casa:
        # Criterio de Kelly
        p = prob_ia 
        q = 1 - p   
        b = cuota_jugada - 1.0 
        
        fraccion_kelly = (p * b - q) / b
        
        # Kelly Fraccional (25%)
        kelly_seguro = fraccion_kelly * 0.25
        
        # Límite: Máximo 5% del bankroll
        if kelly_seguro > 0.05:
            kelly_seguro = 0.05
            
        # Si Kelly da negativo por problemas de redondeo/calibración, no apostamos
        if kelly_seguro <= 0:
            continue
            
        cantidad_por_apuesta = capital_actual * kelly_seguro
        
        apuestas_realizadas += 1
        capital_actual -= cantidad_por_apuesta 
        
        if prediccion_ia == resultado_real:
            ganancia = cantidad_por_apuesta * cuota_jugada
            capital_actual += ganancia
            apuestas_ganadas += 1

roi = ((capital_actual - capital_inicial) / capital_inicial) * 100

print("-" * 30)
print(f"📊 Apuestas realizadas: {apuestas_realizadas} (Muestra robusta)")
print(f"✅ Apuestas ganadas: {apuestas_ganadas}")
if apuestas_realizadas > 0:
    print(f"🎯 Porcentaje de acierto real: {(apuestas_ganadas/apuestas_realizadas)*100:.1f}%")
print(f"💸 Capital Final: {capital_actual:.2f}€")
print(f"📈 ROI (Retorno de Inversión): {roi:.2f}%")