"""
Tests de humo para verificar que los modulos principales del proyecto
son importables y que las dependencias criticas estan instaladas.

Ejecutar con: pytest tests/
"""

import importlib
import pytest


DEPENDENCIAS = [
    "pandas",
    "numpy",
    "sklearn",
    "xgboost",
    "streamlit",
    "google.genai",
    "openpyxl",
    "dotenv",
]


@pytest.mark.parametrize("modulo", DEPENDENCIAS)
def test_dependencia_importable(modulo):
    """Verifica que cada dependencia del proyecto pueda importarse correctamente."""
    mod = importlib.import_module(modulo)
    assert mod is not None, f"El modulo '{modulo}' no pudo importarse."
