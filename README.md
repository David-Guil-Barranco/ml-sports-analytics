# Sistema de Analisis Predictivo Deportivo con Machine Learning

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189BCC?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini%20API-1.0%2B-4285F4?style=flat-square&logo=google&logoColor=white)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green?style=flat-square)

---

## Descripcion

Este repositorio contiene un sistema modular de analisis cuantitativo aplicado a apuestas deportivas. El proyecto aborda el problema de identificar **value bets** —oportunidades en las que la probabilidad real de un resultado supera la probabilidad implicita por las cuotas del mercado— mediante la combinacion de modelos de Machine Learning calibrados y el criterio de gestion de capital de Kelly.

El sistema esta dividido en dos modulos completamente independientes:

| Modulo | Deporte | Fuente de datos | Modelo principal |
|---|---|---|---|
| `laliga` | Futbol (La Liga Espanola) | Football-Data.co.uk (CSV) | Random Forest + Google Gemini API |
| `tenis` | Tenis (Circuito ATP) | JeffSackmann/tennis_atp (CSV) + Bet365 (XLSX) | XGBoost con calibracion Platt Scaling |

### Problema que resuelve

Las casas de apuestas incorporan un margen comercial en sus cuotas que sesga sistematicamente la probabilidad implicita al alza. Un modelo predictivo que estima probabilidades de forma independiente puede detectar las discrepancias entre ambas estimaciones. Cuando la probabilidad predicha supera la probabilidad de equilibrio de la cuota, se genera una value bet con esperanza matematica positiva a largo plazo.

El modulo de **LaLiga** complementa la prediccion estadistica con un analisis de contexto en lenguaje natural generado por el modelo Gemini de Google, aportando factores cualitativos que un modelo de ML puramente cuantitativo no puede capturar.

El modulo de **Tenis** incorpora un simulador financiero completo basado en el criterio de Kelly Fraccional para dimensionar el tamano de cada apuesta en funcion de la ventaja detectada y el capital disponible, con un limite maximo del 5% del bankroll por operacion.

---

## Requisitos previos

- Python 3.11 o superior
- `pip` (gestor de paquetes de Python)
- Una clave de API valida de [Google AI Studio](https://aistudio.google.com/app/apikey) (solo para el modulo LaLiga)
- Git

---

## Instalacion

Sigue estos pasos en orden desde tu terminal.

**1. Clonar el repositorio**

```bash
git clone https://github.com/tu-usuario/predictor-deportivo-ia.git
cd predictor-deportivo-ia
```

**2. Crear y activar el entorno virtual**

```bash
# En Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# En macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

**3. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

**4. Configurar las variables de entorno**

Copia el archivo de plantilla y completa tu clave de API:

```bash
# En Windows (PowerShell)
Copy-Item .env.example .env

# En macOS / Linux
cp .env.example .env
```

Edita el archivo `.env` recien creado y sustituye el valor de `GEMINI_API_KEY` por tu clave real.

---

## Uso

### Modulo LaLiga — Dashboard interactivo (Streamlit)

El modulo principal de LaLiga se ejecuta como una aplicacion web local.

```bash
streamlit run src/laliga/app.py
```

La aplicacion se abrira automaticamente en el navegador en `http://localhost:8501`. Desde la barra lateral es posible seleccionar cualquier partido del dataset activo, ejecutar la prediccion del modelo Random Forest y solicitar el analisis de contexto a Gemini.

### Modulo LaLiga — Simulador por consola (XGBoost, 5 temporadas)

```bash
python src/laliga/simulador.py
```

Ejecuta una simulacion retrospectiva sobre las ultimas 5 temporadas de La Liga empleando un modelo XGBoost con ingenieria de caracteristicas avanzada (medias moviles de goles, tiros, corners y faltas) y el criterio de Kelly Fraccional para la gestion del bankroll.

### Modulo Tenis — Simulador financiero con calibracion

```bash
python src/tenis/bot_final.py
```

Descarga automaticamente los datos historicos del circuito ATP (2019-2024) desde el repositorio publico de Jeff Sackmann, fusiona las estadisticas con las cuotas de Bet365 almacenadas en `data/tenis/`, entrena un modelo XGBoost calibrado con Platt Scaling y ejecuta la simulacion de apuestas sobre la temporada 2024.

### Modulo Tenis — Analisis Over/Under de juegos

```bash
python src/tenis/overunder.py
```

Entrena un regresor XGBoost para predecir el numero total de juegos en un partido ATP y un clasificador para el mercado Over/Under, segmentando los resultados por tramos de cuota para identificar los rangos mas rentables.

---

## Estructura del proyecto

```
.
├── .env.example              # Plantilla de variables de entorno (copiar como .env)
├── .gitattributes            # Politica de saltos de linea para compatibilidad entre SO
├── .gitignore                # Exclusiones de control de versiones
├── .streamlit/
│   ├── config.toml           # Configuracion visual de Streamlit
│   └── secrets.toml.example  # Plantilla de secretos de Streamlit
├── requirements.txt          # Dependencias de Python
├── README.md
│
├── src/                      # Codigo fuente del proyecto
│   ├── laliga/               # Modulo de prediccion de La Liga
│   │   ├── __init__.py
│   │   ├── app.py            # Dashboard principal de Streamlit (Random Forest + Gemini API)
│   │   └── simulador.py      # Simulador retroactivo de 5 temporadas (XGBoost + Kelly)
│   │
│   └── tenis/                # Modulo de prediccion del circuito ATP
│       ├── __init__.py
│       ├── bot_final.py      # Pipeline completo: descarga ATP + modelo calibrado + simulacion
│       ├── kelly.py          # Simulador financiero con Kelly Fraccional (datos locales XLSX)
│       ├── modelo.py         # XGBoost con historial de victorias por superficie
│       └── overunder.py      # Predictor de total de juegos (mercado Over/Under)
│
├── data/                     # Datasets
│   ├── laliga/               # Temporadas La Liga Primera y Segunda Division (CSV)
│   │   ├── SP1_21-22.csv
│   │   ├── ...
│   │   └── SP2_25-26.csv
│   │
│   └── tenis/                # Cuotas ATP por temporada (formato Bet365 XLSX)
│       ├── tenis_2023.xlsx
│       └── tenis_2024.xlsx
│
└── tests/
    └── test_smoke.py         # Tests de humo: verifica que las dependencias son importables
```

---

## Tecnologias y librerias

| Tecnologia | Uso en el proyecto |
|---|---|
| **Python 3.11** | Lenguaje base del proyecto |
| **Streamlit** | Interfaz web interactiva del modulo LaLiga |
| **pandas** | Manipulacion y transformacion de datos tabulares |
| **NumPy** | Operaciones vectoriales y generacion de variables sinteticas |
| **scikit-learn** | Random Forest, calibracion (Platt Scaling), metricas de evaluacion |
| **XGBoost** | Clasificacion y regresion de alta capacidad para los modelos principales |
| **Google Gemini API** | Analisis de contexto cualitativo en lenguaje natural |
| **openpyxl** | Lectura de archivos Excel (.xlsx) con cuotas de Bet365 |
| **python-dotenv** | Gestion segura de variables de entorno |

---

## Datos

### Fuentes externas

- **Football-Data.co.uk**: Historico de partidos de La Liga (primera y segunda division) en formato CSV. Incluye cuotas de Bet365 (B365H, B365D, B365A), goles, tiros, corners y faltas.
- **JeffSackmann/tennis_atp**: Repositorio publico en GitHub con estadisticas detalladas de todos los partidos del circuito ATP desde 1968.

### Datos locales incluidos

Los archivos CSV de La Liga (`data/laliga/`) y los XLSX de tenis (`data/tenis/`) estan incluidos en el repositorio para facilitar la reproducibilidad. Si los datos fueran privados o de tamano excesivo, las rutas correspondientes deberian anadirse al `.gitignore`.

---

## Seguridad

**Advertencia importante**: Nunca subas tu clave de API de Gemini al repositorio. Este proyecto emplea variables de entorno a traves de un archivo `.env` que esta explicitamente excluido en el `.gitignore`. La clave de API es una credencial privada equivalente a una contrasena.

Si detectas que una clave fue expuesta en el historial de Git, revocala inmediatamente desde [Google AI Studio](https://aistudio.google.com/app/apikey) y genera una nueva.

---

## Licencia

Distribuido bajo la licencia MIT. Consulta el archivo `LICENSE` para mas informacion.
