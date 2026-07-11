import pandas as pd

print("🧬 INICIANDO EL LABORATORIO: LA GRAN FUSIÓN DE DATOS (2024)...\n")

# --- 1. CARGAMOS LAS DOS BASES DE DATOS ---
print("📥 Descargando estadísticas detalladas desde GitHub...")
df_stats = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2024.csv")

print("📥 Leyendo cuotas locales desde 2024.xlsx...")
try:
    df_odds = pd.read_excel("data/tenis/tenis_2024.xlsx")
except FileNotFoundError:
    print("❌ Error: No encuentro el archivo '2024.xlsx'.")
    exit()

# --- 2. EL TRUCO DEL APELLIDO ---
print("⚙️ Estandarizando nombres de jugadores...")
# Sackmann: "Carlos Alcaraz" -> Nos quedamos la última palabra en minúsculas -> "alcaraz"
df_stats['w_apellido'] = df_stats['winner_name'].astype(str).apply(lambda x: x.split()[-1].lower())
df_stats['l_apellido'] = df_stats['loser_name'].astype(str).apply(lambda x: x.split()[-1].lower())

# Excel: "Alcaraz C." -> Nos quedamos la primera palabra en minúsculas -> "alcaraz"
df_odds['w_apellido'] = df_odds['Winner'].astype(str).apply(lambda x: x.split()[0].lower())
df_odds['l_apellido'] = df_odds['Loser'].astype(str).apply(lambda x: x.split()[0].lower())

# --- 3. EL TRUCO DEL MES ---
print("📅 Sincronizando calendarios...")
# Sackmann usa el formato YYYYMMDD numérico (ej. 20240514) -> Extraemos el mes
df_stats['mes'] = (df_stats['tourney_date'] // 100) % 100

# Excel usa fechas reales -> Extraemos el mes
df_odds['mes'] = pd.to_datetime(df_odds['Date']).dt.month

# --- 4. LA MAGIA: EL MERGE ---
print("🔗 Cruzando bases de datos...")
# Unimos las filas donde coincidan ganador, perdedor y el mes en el que jugaron
df_fusion = pd.merge(df_stats, df_odds, on=['w_apellido', 'l_apellido', 'mes'], how='inner')

# Por si jugaron dos veces en el mismo mes (raro, pero posible), borramos duplicados exactos
df_fusion = df_fusion.drop_duplicates(subset=['tourney_name', 'w_apellido', 'l_apellido'])

print("\n" + "="*40)
print("🏆 RESULTADOS DE LA FUSIÓN 🏆")
print("="*40)
print(f"Partidos con estadísticas (Sackmann): {len(df_stats)}")
print(f"Partidos con cuotas (Excel):          {len(df_odds)}")
print(f"✅ PARTIDOS FUSIONADOS CON ÉXITO:     {len(df_fusion)}")
print("="*40)

print("\n🔍 Vistazo a tu nueva Súper-Tabla (Estadísticas + Cuotas):")
# Mostramos que ahora tenemos los primeros saques (stats) Y las cuotas (odds) juntos
columnas_estrella = ['winner_name', 'loser_name', 'w_1stIn', 'B365W', 'B365L']
print(df_fusion[columnas_estrella].head())