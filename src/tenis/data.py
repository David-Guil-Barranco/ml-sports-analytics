import pandas as pd

print("🎾 DESCARGANDO DATOS HISTÓRICOS DE LA ATP...")

# Descargamos los datos del repositorio oficial de Jeff Sackmann directamente desde GitHub
url_2023 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
url_2024 = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv"

# Leemos y unimos los años
df_2023 = pd.read_csv(url_2023)
df_2024 = pd.read_csv(url_2024)
df_tenis = pd.concat([df_2023, df_2024], ignore_index=True)

# Filtramos las columnas más interesantes para empezar
# winner/loser = ganador/perdedor
# rank = ranking ATP
# w_1stIn / l_1stIn = % de primeros saques dentro
# w_bpSaved / l_bpSaved = Puntos de rotura salvados
columnas_clave = [
    'tourney_name', 'surface', 'winner_name', 'loser_name', 
    'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
    'w_1stIn', 'l_1stIn', 'w_bpSaved', 'l_bpSaved'
]

df_limpio = df_tenis[columnas_clave].dropna().copy()

print(f"✅ ¡Datos descargados! Tenemos {len(df_limpio)} partidos profesionales con estadísticas completas.\n")

print("📊 Vistazo a los 3 últimos partidos registrados:")
print(df_limpio[['tourney_name', 'surface', 'winner_name', 'loser_name']].tail(3))