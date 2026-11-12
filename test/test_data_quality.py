import pandas as pd
import pytest
from pandera.pandas import DataFrameSchema, Column, Check
from sqlalchemy import column
import warnings

GENEROS = ["Femenino", "Masculino", "Otro"]
SENTIMIENTOS = ["Positivo", "Negativo", "Neutral"]

@pytest.fixture

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

def test_schema_datos_clientes (datos_clientes):
    """ Función para validar el esquema de los datos de clientes

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes

     return none
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
    # Lazy=True permite que se muestren todos los errores de validación en lugar de detenerse en el primer error encontrado.
    schema.validate(datos_clientes, lazy=True)

def test_schema_datos_resenas (datos_resenas):
    """ Función para validar el esquema de los datos de reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    return none
    """

    schema = DataFrameSchema(
        columns={
            "id_resena": Column(int, nullable=False),
            "id_cliente": Column(int, nullable=False),
            "fecha_resena": Column(str, nullable=True),
            "calificacion": Column(int, checks=Check.in_range(1,5), nullable=False),
            "sentimiento": Column(str, checks=Check.isin(SENTIMIENTOS), nullable=False),

            # "comentario": Column(str, nullable=False),
            # TODO: agregar la validación de los atributos restantes.
        }
    )
    schema.validate(datos_resenas, lazy=True)

def test_basicos_clientes(datos_clientes):
    """ Función para validar los datos basicos de los clientes

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes

    return none
    """

    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_clientes.empty, "El dataframe de clientes está vacío"
    assert datos_clientes.shape[1] == 6, "El dataframe de clientes no tiene 6 columnas"
    assert datos_clientes["id_cliente"].notnull().all(), "Existen valores nulos en la columna id_cliente"
    
    # Extra: Pueden completar otras validaciones básicas.
    assert datos_clientes["genero"].notnull().all(), "Existen valores nulos en la columna genero"
    assert datos_clientes["ciudad"].notnull().all(), "Existen valores nulos en la columna ciudad"
    assert datos_clientes["antiguedad_cliente_meses"].notnull().all(), "Existen valores nulos en la columna antiguedad_cliente_meses"
    assert datos_clientes["canal_registro"].notnull().all(), "Existen valores nulos en la columna canal_registro"


def test_basicos_resenas(datos_resenas):
    """ Función para validar los datos basicos de las reseñas

    Args:
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    return none
    """

    # Validar que no existan valores nulos en las columnas obligatorias
    assert not datos_resenas.empty, "El dataframe de reseñas está vacío"
    assert datos_resenas.shape[1] == 13, "El dataframe de reseñas no tiene 13 columnas"
    assert datos_resenas["id_resena"].notnull().all(), "Existen valores nulos en la columna id_resena"
    assert datos_resenas["id_cliente"].notnull().all(), "Existen valores nulos en la columna id_cliente"
    

    # Extra: Pueden completar otras validaciones básicas.
    assert datos_resenas["calificacion"].notnull().all(), "Existen valores nulos en la columna calificacion"
    assert datos_resenas["sentimiento"].notnull().all(), "Existen valores nulos en la columna sentimiento"
    assert datos_resenas["fecha_resena"].notnull().all(), "Existen valores nulos en la columna fecha_resena"

def test_integridad_referencial(datos_clientes, datos_resenas):
    """ Función para validar la integridad referencial entre los datos de clientes y reseñas

    Args:
        datos_clientes (pd.DataFrame): Dataframe con los datos de clientes
        datos_resenas (pd.DataFrame): Dataframe con los datos de reseñas

    return none
    """

    # Validar que todos los id_cliente en datos_resenas existan en datos_clientes
    df_ids = datos_resenas[['id_cliente']].merge(datos_clientes[['id_cliente']], 
            on='id_cliente', 
            how='left', 
            indicator=True) # genera la columna _merge con el resultado de la unión

    df_ids["_merge"] = df_ids["_merge"].astype(str)

    integridad_referencial = DataFrameSchema(columns={
        "_merge": Column(str,nullable=False, checks=Check.isin(["both"]))
    }
    )
    integridad_referencial.validate(df_ids, lazy=True)

####
# TODO: Agregar al menos una función de test con una o más validaciones más allá de las estructuras básicas de cada dataset.
# Por ejemplo: unicidad de claves (id) y completar las evaluaciones de estructuras pendientes (reseñas).

def test_unicidad_ids(datos_clientes, datos_resenas):
    """Valida que las claves primarias sean únicas"""

    schema_clientes = DataFrameSchema(
            columns={
                "id_cliente": Column(
                    int,
                    unique=True,
                    nullable=False
                )
            }
        )

    schema_resenas = DataFrameSchema(
            columns={
                "id_resena": Column(
                    int,
                    unique=True,
                    nullable=False
                )
            }
        )

    schema_clientes.validate(datos_clientes, lazy=True)
    schema_resenas.validate(datos_resenas, lazy=True)

# Regla de negocio: edad positiva, antigüedad no negativa, calificación entre 1 y 5
def test_reglas_negocio(datos_clientes, datos_resenas):

    # edad positiva
    assert (datos_clientes["edad"].dropna() >= 18).all(), "Existen clientes menores de edad"

    # antigüedad no negativa
    assert (
        datos_clientes["antiguedad_cliente_meses"] >= 0
    ).all(), "Existen clientes con antigüedad negativa"

    # calificación entre 1 y 5
    assert (
        datos_resenas["calificacion"]
            .between(1, 5)
            .all(), "Existen reseñas con calificación fuera del rango 1-5"
    )

# Extra: validar alguna regla de negocio como podría ser la consistencia entre la calificación y sentimiento entendimiento que 
# una califificación baja se relaciona con un sentimiento negativo y una calificación alta con un sentimiento positivo.
# Por ejmplo, si la calificación es menor a 3, el sentimiento no debería ser positivo en una proporción mayor al 5%.
# Alerta para que alguien experto en el dominio lo revise.

def test_consistencia_calificacion_sentimiento(datos_resenas):
    """
    Regla de negocio:
    Si la calificación es menor a 3, el sentimiento positivo
    no debería superar el 5%.
    """
    # Seleccionar reseñas con calificación baja
    resenas_baja_calificacion = datos_resenas[
        datos_resenas["calificacion"] < 3
    ]

    # Evitar división por cero si no existen registros
    if len(resenas_baja_calificacion) == 0:
        return

    # Contar reseñas con sentimiento positivo
    positivas = len(
        resenas_baja_calificacion[
            resenas_baja_calificacion["sentimiento"] == "Positivo"
        ]
    )

    # Calcular proporción de inconsistencias
    porcentaje_positivas = (
        positivas / len(resenas_baja_calificacion)
    )

   # Generar alerta sin fallar el test
    if porcentaje_positivas > 0.05:
        warnings.warn(
            f"ALERTA DE CALIDAD: "
            f"{porcentaje_positivas:.2%} de las reseñas con "
            f"calificación menor a 3 tienen sentimiento positivo. "
            f"Se recomienda revisión por un experto del dominio."
        )