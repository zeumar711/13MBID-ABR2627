# Se importan las librerías necesarias y se suprimen las advertencias
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from dotenv import load_dotenv, find_dotenv
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

REGISTERED_MODEL_NAME = "sentimiento-resenas"
ALIAS_PRODUCCION = "champion"
UMBRAL_ACCURACY = 0.8

def load_data(data_path: str = "data/processed/datos_integrados.csv"):
    """
    Carga los datos desde un archivo CSV y devuelve un DataFrame de pandas.
    
    Args:
        path (str): Ruta del archivo CSV.
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    # Lectura de los datos
    df = pd.read_csv(data_path)
    target = "sentimiento"

    features_X = df.drop(columns=[target])
    labels_y = df[target]

    # Primero se separa el test final (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_X,
        labels_y,
        test_size=0.10,
        random_state=42,
        stratify=labels_y
    )

    # Luego se separan train y validation (22% del 90% es aprox. el 20% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.22,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y

def create_preprocessor(features_X: pd.DataFrame):
    """
    Crea un preprocesador para los datos de entrada.
    
    Args:
        features_X (pd.DataFrame): DataFrame con las características de entrada.
        
    Returns:
        ColumnTransformer: Preprocesador para los datos de entrada.
    """
    # Identificación de columnas numéricas y categóricas
    numeric_features = features_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = features_X.select_dtypes(include=['object','string']).columns.tolist()

    # Definición de transformadores para cada tipo de columna
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    text_col = "texto_resena"
    text_transformer = TfidfVectorizer(max_features=300)

    # Creación del preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_col)  
        ]
    )

    return preprocessor

def train_model(
    data_path: str = "data/processed/datos_integrados.csv",
    model_output_path: str = "models/trained_model.pkl",
    preprocessor_output_path: str = "models/preprocessor.pkl",
    metrics_output_path: str = "models/train_metrics.json",
    figures_output_path: str = "docs/figures",
    registered_model_name: str | None = REGISTERED_MODEL_NAME,
):

        """
        Entrena un modelo de clasificación utilizando los datos procesados y guarda el modelo entrenado, el preprocesador y las métricas de evaluación.

        Args:
            data_path (str): Ruta del archivo CSV con los datos procesados.
            model_output_path (str): Ruta donde se guardará el modelo entrenado.
            preprocessor_output_path (str): Ruta donde se guardará el preprocesador.
            metrics_output_path (str): Ruta donde se guardará las métricas de evaluación.
            figures_output_path (str): Ruta donde se guardarán las figuras de evaluación.
        """
    
        # Recordar que debe existir un archivo .env en la raíz del proyecto con las variables 
        # de entorno necesarias para MLflow.
        # Las credenciales de MLflow se pueden obtener en Dagshub, en la sección de 
        # Settings > Tokens > New Token o copiando el existente.
        load_dotenv(find_dotenv())  # Carga las variables de entorno para MLflow
        mlflow.set_experiment("Proyecto 13MBID - Entrenamiento de modelos de clasificación de sentimientos")
        
        # Carga de los datos
        X_train, X_val, X_test, y_train, y_val, y_test, features_X, labels_y = load_data(data_path)

        # Carga del preprocesador
        preprocessor = create_preprocessor(features_X)

        # Definición de los modelos a entrenar
        model = LogisticRegression(max_iter=1000)

        # Creación del pipeline que incluye el preprocesador y el modelo
        pipeline = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", model)])

        # Entrenamiento del modelo
        pipeline.fit(X_train, y_train)

        # Evaluación del modelo en el conjunto de test
        y_test_pred = pipeline.predict(X_test)
        y_test_proba = pipeline.predict_proba(X_test)

        # Cálculo de métricas de evaluación
        metrics = {
        "test_accuracy":accuracy_score(y_test, y_test_pred),
        "test_precision_macro":precision_score(y_test, y_test_pred, average="macro",zero_division = 0),
        "test_recall_macro":recall_score(y_test, y_test_pred, average="macro",zero_division = 0),
        "test_f1_macro":f1_score(y_test, y_test_pred, average="macro",zero_division = 0),
        "test_recall_negative":recall_score(y_test, y_test_pred, pos_label="negativo", average="macro", zero_division=0),
        "test_roc_aux_ovr":roc_auc_score(y_test, y_test_proba, multi_class="ovr", labels=pipeline.classes_),

        }

        for metric_name, metric_value in metrics.items():
            print(f"[{metric_name} {metric_value:.4f}")

        # Guardadado de la matriz de confusión
        Path(figures_output_path).mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_test, y_test_pred, labels=pipeline.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Logistic Regression - Test Set")
        cm_fig_path = f"{figures_output_path}/confusion_matrix_test.png"
        plt.savefig(cm_fig_path)
        plt.close()

        # Registro general en MLFlow
        signature = infer_signature(X_train, pipeline.predict(X_train))

        with mlflow.start_run(run_name="Pipeline (prod) - Logistic Regression") as run:
            mlflow.log_params(model.get_params())
            mlflow.log_params ({
                "train_sample_size": len(X_train),
                "validation_sample_size": len(X_val),
                "test_sample_size": len(X_test),
                "tfidf_max_festures":300,
            })

            mlflow.log_metrics(metrics)
            mlflow.log_artifact(cm_fig_path, artifact_path="figures")
            mlflow.sklearn.log_model(pipeline, name="model",
                signature=signature, input_example=X_train.head(3))

            run_id = mlflow.active_run().info.run_id
            print(f"Modelo registrado en MLFlow a raíz del run ID:{run_id}")

        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardado del modelo entrenado y del preprocesador
        joblib.dump(pipeline,model_output_path)
        joblib.dump(pipeline.named_steps["prep"],preprocessor_output_path)

        # Guardado de métricas en un archivo JSON
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics,f, indent=2)

        return pipeline, pipeline.named_steps['prep'], metrics

if __name__ == "__main__":
    train_model()

# TODO: Seleccionar el mejor modelo entre los candidatos basados en otras métricas
# en lugar de solo accuracy y registrar el mejor modelo en MLFlow, utilizando este scripts.
# Justificar la elección de las métricas y el modelo.

# Incorporar la imputación de valores faltantes dentro del pipeline para que el modelo
# pueda recibir los datos sin procesarlos previamente en la instancia de producción.

# Extra: Utilizar alguna técnica diferent, revisar hiperparametrización, considerar
# alternativas para el desbalanceo de las clases
