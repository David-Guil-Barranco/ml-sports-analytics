import os
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from google import genai
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURACION DE LA PAGINA ---
st.set_page_config(page_title="IA Tipster - Predictor Deportivo", layout="wide")
st.title("Dashboard: Analista Deportivo IA")
st.markdown("Combina el rigor matematico de un **Random Forest** con el analisis de contexto de **Gemini Flash**.")

# --- CONFIGURACION DE GEMINI ---
# La clave se carga desde la variable de entorno GEMINI_API_KEY (archivo .env)
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key:
    st.error("Variable de entorno GEMINI_API_KEY no configurada. Consulta el archivo .env.example.")
    st.stop()
client = genai.Client(api_key=_api_key)

# --- 1. CARGA DE DATOS Y MODELO (Caché para que sea rápido) ---
@st.cache_data
def preparar_modelo():
    # Ruta relativa desde la raiz del proyecto (ejecutar con: streamlit run src/laliga/app.py)
    df = pd.read_csv('data/laliga/SP1_25-26.csv')
    df_limpio = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']].copy()
    df_limpio['Target'] = df_limpio['FTR'].map({'H': 1, 'D': 0, 'A': 2})
    df_limpio['Ultimo_Resultado_Casa'] = df_limpio.groupby('HomeTeam')['Target'].shift(1)
    df_modelo = df_limpio.dropna()
    
    # Entrenamos el modelo con todos los datos disponibles
    X = df_modelo[['B365H', 'B365D', 'B365A', 'Ultimo_Resultado_Casa']]
    y = df_modelo['Target']
    modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    modelo.fit(X, y)
    return df_modelo, modelo

df_modelo, modelo_bosque = preparar_modelo()

# --- 2. INTERFAZ DE USUARIO ---
st.sidebar.header("Selecciona un Partido")
# Creamos una lista de partidos únicos para el desplegable
lista_partidos = (df_modelo['HomeTeam'] + " vs " + df_modelo['AwayTeam']).unique()
partido_elegido = st.sidebar.selectbox("Partido a analizar:", lista_partidos)

if st.sidebar.button("Analizar con IA 🚀"):
    # Extraemos los equipos
    local, visitante = partido_elegido.split(" vs ")
    
    # Buscamos las cuotas promedio de ese partido en nuestro dataset
    datos_partido = df_modelo[(df_modelo['HomeTeam'] == local) & (df_modelo['AwayTeam'] == visitante)].iloc[-1]
    
    cuota_1 = datos_partido['B365H']
    cuota_X = datos_partido['B365D']
    cuota_2 = datos_partido['B365A']
    ultimo_res = datos_partido['Ultimo_Resultado_Casa']
    
    # Hacemos la predicción matemática
    prediccion_num = modelo_bosque.predict([[cuota_1, cuota_X, cuota_2, ultimo_res]])[0]
    probabilidades = modelo_bosque.predict_proba([[cuota_1, cuota_X, cuota_2, ultimo_res]])[0]
    
    diccionario_texto = {1: "Victoria Local", 0: "Empate", 2: "Victoria Visitante"}
    
    # --- 3. MOSTRAR RESULTADOS MATEMÁTICOS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Cuotas Bet365", f"1: {cuota_1} | X: {cuota_X} | 2: {cuota_2}")
    col2.metric("Predicción Estadística", diccionario_texto[prediccion_num])
    col3.metric("Confianza del Modelo", f"{max(probabilidades)*100:.1f}%")
    
    st.divider()
    
    # --- 4. ANÁLISIS DE GEMINI ---
    st.subheader("🤖 Análisis de Contexto por Gemini")
    
    prompt = f"""
    Eres un tipster profesional. Mi algoritmo Random Forest predice '{diccionario_texto[prediccion_num]}' 
    para el partido {local} vs {visitante}. 
    Las cuotas son: Local {cuota_1}, Empate {cuota_X}, Visitante {cuota_2}. 
    El algoritmo tiene una confianza del {max(probabilidades)*100:.1f}%.
    
    Redacta un análisis de 3 párrafos cortos:
    1. ¿Tiene sentido estadístico esta predicción basándote en las cuotas?
    2. Nombra un factor de contexto real (histórico, táctico o de entrenador) de estos equipos que el modelo no esté viendo.
    3. Tu veredicto final: ¿Apostarías a esto o es un riesgo?
    """
    
    with st.spinner('Gemini está analizando el contexto deportivo...'):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt,
            )
            st.success("Análisis completado.")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error al conectar con la IA: {e}")