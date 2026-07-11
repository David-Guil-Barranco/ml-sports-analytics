import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
df_laliga = pd.read_csv('SP1 25-26.csv')
columnas_clave = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
df_limpio = df_laliga[columnas_clave].copy()

# Mapeamos letras a números
diccionario_resultados = {'H': 1, 'D': 0, 'A': 2}
df_limpio['Target'] = df_limpio['FTR'].map(diccionario_resultados)

# --- 2. INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING) ---
# Aquí es donde creamos la columna para que el modelo tenga contexto
df_limpio['Ultimo_Resultado_Casa'] = df_limpio.groupby('HomeTeam')['Target'].shift(1)

# Borramos los partidos de la jornada 1 que no tienen historial (los NaN)
df_modelo = df_limpio.dropna()

print("Datos limpios y con historial listos.")

# --- 3. MACHINE LEARNING: RANDOM FOREST ---
# Separamos pistas (X) y objetivo (y)
X = df_modelo[['B365H', 'B365D', 'B365A', 'Ultimo_Resultado_Casa']]
y = df_modelo['Target']

# Repartimos 80% estudiar / 20% examinar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Invocamos al Random Forest (100 árboles)
modelo_bosque = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Entrenamos y predecimos
modelo_bosque.fit(X_train, y_train)
predicciones_bosque = modelo_bosque.predict(X_test)

# Vemos la nota final
precision_bosque = accuracy_score(y_test, predicciones_bosque)
print(f"Precisión del Random Forest: {precision_bosque * 100:.2f}%")