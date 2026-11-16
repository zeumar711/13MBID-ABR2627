import json
from pathlib import Path

import pytest

from src.train_model import train_model

# Objetivo de negocio: 80% de precisión en el conjunto de prueba
TARGET_ACCURACY = 0.80


def test_train_model(tmp_path):
    # Configurar acceso a los archivos
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "metrics" / "train_metrics.json"

    if not baseline_path.exists():
        pytest.skip(f"Archivo de referencia {baseline_path} no encontrado. Se omite la prueba.")

    # Cargar métricas de referencia
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Ejecutar la función de entrenamiento del modelo
    data_path = project_root / "data" / "processed" / "datos_integrados.csv"
    model_output_path = tmp_path / "prod_model.pkl"
    preprocessor_output_path = tmp_path / "prod_preprocessor.pkl"
    metrics_output_path = tmp_path / "train_metrics.json"

    _, _, metrics = train_model(
        data_path=str(data_path),
        model_output_path=str(model_output_path),
        preprocessor_output_path=str(preprocessor_output_path),
        metrics_output_path=str(metrics_output_path),
        figures_output_path=str(tmp_path),
    )

    # Se corroboran las métricas generadas
    assert set(metrics.keys()) == set(baseline.keys()), "Las métricas generadas no coinciden con las métricas de referencia."

    atol = 1e-9  # Tolerancia absoluta para la comparación de métricas
    for k in baseline.keys():
        assert metrics[k] == pytest.approx(baseline[k], rel=0, abs=atol), f"La métrica '{k}' no coincide con la referencia."


def test_train_model_accuracy_threshold():
    # Configurar acceso a los archivos
    project_root = Path(__file__).resolve().parents[1]
    metrics_path = project_root / "metrics" / "train_metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"Archivo de métricas {metrics_path} no encontrado. Se omite la prueba.")
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert metrics["test_accuracy"] >= TARGET_ACCURACY, f"La precisión del modelo en el conjunto de prueba es menor que el umbral objetivo de {TARGET_ACCURACY}."


#Extra: agregar alguna verificación adicional del modelo con recall u otro factor
#       También se puede verificar si el modelo obtenido genera resultados mejores que
#       un modelo base (baseline) como un DummyClassifier, por ejemplo.


if __name__ == "__main__":
    test_train_model()
    test_train_model_accuracy_threshold()