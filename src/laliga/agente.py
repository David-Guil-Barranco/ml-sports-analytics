import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

print("--- CONECTANDO CON GEMINI ---")

# Configurar el cliente con la clave cargada desde .env
_api_key = os.getenv("GEMINI_API_KEY", "")
if not _api_key:
    raise EnvironmentError("Variable de entorno GEMINI_API_KEY no configurada. Consulta el archivo .env.example.")
client = genai.Client(api_key=_api_key)

# 2. Le damos contexto y un rol (El Prompt)
mensaje = """
Eres un analista de apuestas de fútbol experto y directo. 
Acabo de correr un modelo Random Forest para el próximo partido entre 
Real Madrid y Getafe en el Bernabéu. El modelo me da un 47.2% de probabilidad 
de victoria local.

Sabiendo esto, redacta un comentario muy breve (máximo 3 líneas) diciendo si te parece 
una predicción lógica y qué factor sorpresa podría estropear la apuesta.
"""

print("Pensando...\n")

# 3. Hacemos la llamada al nuevo modelo (usamos la versión 1.5 Flash, que es rapidísima)
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents=mensaje,
)

# 4. Mostramos la respuesta
print("🤖 RESPUESTA DEL AGENTE:")
print(response.text)