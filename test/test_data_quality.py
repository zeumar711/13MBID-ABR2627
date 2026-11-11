import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check

GENEROS = ["Femenino", "Masculino", "Otro"]
SENTIMIENTOS = ["Positivo", "Negativo", "Neutral"]

def datos_clientes():
    """ Función para cargar los datos de clientes en un dataframe
    Args:
        None
    Returns:
        pd.DataFrame: Dataframe con los datos de clientes
    """

    df=pd.read_csv("data/raw/datos_clientes.csv", sep=";")
    return df

@pytest.fixture

def datos_resenas():
    """ Función para cargar los datos de reseñas en un dataframe
    Args:
        None
    Returns:
        pd.DataFrame: Dataframe con los datos de reseñas
    """

    df=pd.read_csv("data/raw/datos_resenas.csv", sep=";")
    return df

def test_schema_datos_clientees (datos_clientes):
    """ Función para validar el esquema de los datos de clientes

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
    """

    schema = DataFrameSchema(
        columns={
            "id_cliente": Column(int, nullable=False),
            "edad": Column(float, checks=Check.in_range(18,90), nullable=True),
            "genero": Column(str, checks=Check.isin(GENEROS), nullable=False),
            "ciudad": Column(str, nullable=False),
            "antiguedad_cliente_meses": Column(int, checks=Check.ge(0), nullable=True),
            "canal_registro": Column(str, nullable=False),
        }
    )
    schema.validate(datos_clientes)

def test_schema_datos_resenas (datos_resenas):
    """ Función para validar el esquema de los datos de reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas
    """

    schema = DataFrameSchema(
        columns={
            "id_resena": Column(int, nullable=False),
            "id_cliente": Column(int, nullable=False),
            "fecha_resena": Column(str, nullable=True),
            "calificacion": Column(float, checks=Check.in_range(1,5), nullable=False),
            "sentimiento": Column(str, checks=Check.isin(SENTIMIENTOS), nullable=False),

            # "comentario": Column(str, nullable=False),
            # TODO: agregar la validación de los atributos restantes.
        }
    )
    schema.validate(datos_resenas)