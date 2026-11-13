# Importación de librerías y supresión de advertencias
import pandas as pd
from pathlib import Path

def process_data(
    datos_clientes: str= "data/raw/datos_clientes.csv",
    datos_resenas: str= "data/raw/datos_resenas.csv",
    output_dir: str= "data/processed",
) -> None:
    
    """
    Función para procesar los datos de clientes y reseñas

    Args:
        datos_clientes (str): Ruta al archivo CSV de datos de clientes. 
        datos_resenas (str): Ruta al archivo CSV de datos de reseñas.
        output_dir (str): Ruta al directorio donde se guardarán los archivos procesados.
    Returns:
        None
    """
    # lectura de los datos
    df_clientes = pd.read_csv(datos_clientes,sep=";")
    df_resenas = pd.read_csv(datos_resenas,sep=";")
   
   # Selección de los datos (filtro de columnas)
   # TODO: Agregar si corresponde otras acciones en función de la evaluación de calidad de los datos del paso previo.
    col_eliminar_clientes = ['canal_registro']
    df_clientes.drop(col_eliminar_clientes, inplace=True, axis=1)

    col_eliminar_resenas = ['fecha_resena','reclamo_formal']
    df_resenas.drop(col_eliminar_resenas, inplace=True, axis=1)

    # Limpieza de los datos (filtro de filas)
    df_clientes_filtrado = df_clientes[(df_clientes['edad'] < 90) | (df_clientes['edad'].isna())]
    df_resenas_filtrado = df_resenas.drop_duplicates().copy()

    # Tratamiento de valores nulos 
    df_clientes_filtrado['edad'] = df_clientes_filtrado.groupby("ciudad")["edad"]\
                        .transform(lambda x: x.fillna(x.median()))

    # Tratamiento de nulos para tiempo_respuesta_horas (se imputa con la mediana según categoría de producto)
    df_resenas_filtrado['tiempo_respuesta_horas'] = df_resenas_filtrado.groupby("categoria_producto")["tiempo_respuesta_horas"]\
                       .transform(lambda x: x.fillna(x.median()))

    # Eliminación de llaves duplicadas en la columna 'id_cliente' y 'id_resena'
    # 1. Eliminación de duplicados en clientes (Llave: 'id_cliente')
    filas_clientes_antes = len(df_clientes_filtrado)
    df_clientes_filtrado = df_clientes_filtrado.drop_duplicates(subset=['id_cliente'], keep='first')

    duplicados_clientes = filas_clientes_antes - len(df_clientes_filtrado)
    print(f'Clientes duplicados eliminados: {duplicados_clientes}')

    # 2. Eliminación de duplicados en reseñas (Llave: 'id_resena')
    filas_resenas_antes = len(df_resenas_filtrado)
    df_resenas_filtrado = df_resenas_filtrado.drop_duplicates(subset=['id_resena'], keep='first')

    duplicados_resenas = filas_resenas_antes - len(df_resenas_filtrado)
    print(f'Reseñas duplicadas eliminadas: {duplicados_resenas}')

    # 3. Verificación de unicidad sobre los DataFrames filtrados
    assert df_clientes_filtrado['id_cliente'].is_unique, 'Error: Aún hay id_cliente duplicados'
    assert df_resenas_filtrado['id_resena'].is_unique, 'Error: Aún hay id_resena duplicados'
    print('Validación exitosa: Ambas llaves primarias son 100% únicas.')

    # Integración de los datos
    df_integrado = pd.merge(
        df_resenas_filtrado, 
        df_clientes_filtrado, 
        on='id_cliente', 
        how='inner')

    # Creación de atributos nuevos

    # Longitud del texto de la reseña, en caracteres
    df_integrado["longitud_resena"] = df_integrado["texto_resena"].str.len()

    # Cantidad de palabras del texto de la reseña
    df_integrado["cantidad_palabras_resena"] = df_integrado["texto_resena"].str.split().apply(len)

    # Tiempo de respuesta expresado en días, más interpretable que en horas
    df_integrado["tiempo_respuesta_dias"] = df_integrado["tiempo_respuesta_horas"] / 24

    # Se creó un campo que incluya la validación de 3 reglas de negocio.
    # # 1. Definimos cada condición por separado para mayor claridad y mantenimiento
    condicion_mayor_edad = (df_integrado['edad'] >= 18) | (df_integrado['edad'].isna())  # Permite nulos sin falso positivo
    condicion_antiguedad = df_integrado['antiguedad_cliente_meses'] >= 0             # Antigüedad mayor o igual a 0
    condicion_calificacion = df_integrado['calificacion'].between(1, 5)              # Rango inclusivo entre 1 y 5

    # 2. Creamos la columna booleana (True = Cumple todas las reglas / False = Incumple alguna)
    df_integrado['cumple_reglas_negocio'] = (
        condicion_mayor_edad & 
        condicion_antiguedad & 
        condicion_calificacion
)
    df_integrado['cumple_reglas_negocio'] = df_integrado['cumple_reglas_negocio'].astype(int)

    # Se creo un campo que incluya otra regla de negocio. 
    # Crea la columna booleana: True si incumple la regla (calificación < 3 y sentimiento 'Positivo')
    df_integrado["inconsistencia_sentimiento"] = (
    df_integrado["calificacion"] < 3
) & (df_integrado["sentimiento"] == "Positivo").astype(int)

    # TODO: Otros atributos que pueden generarse, (implementar dos al menos)
    # por ejemplo:
    # - Cantidad de reseñas por cliente
    # - Promedio de calificación por cliente
    
    # 1. Cantidad total de reseñas por cada cliente
    df_integrado['cantidad_resenas_cliente'] = df_integrado.groupby('id_cliente')['id_resena'].transform('count')

    # 2. Promedio de calificación histórica por cada cliente
    df_integrado['promedio_calificacion_cliente'] = df_integrado.groupby('id_cliente')['calificacion'].transform('mean')

    # Fomateo de los datos

    # Se eliminan columnas identificadoras, la fecha (no se utiliza en esta etapa)
    # y 'calificacion', dado que determina directamente a 'sentimiento' y generaría data leakage
    columnas_a_eliminar = [
    "id_cliente",
    "id_resena",
    "calificacion",
]
    df_final = df_integrado.drop(columns=columnas_a_eliminar)

    # Generar el csv de salida

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    df_final.to_csv(f"{output_dir}/datos_integrados.csv", index=False)


    # Se verifica la cantidad de valores nulos en cada columna del dataset final
    print("Valores nulos en el dataset final:")
    for col in df_final.columns:
        print(f"Atributo: {col} - Valores nulos: {df_final[col].isna().sum()} ({df_final[col].isna().sum()/len(df_final)*100:.2f}%)")

    # Se verifica la presencia de valores duplicados en el dataset final
    num_duplicados = df_final.duplicated().sum()
    print(f"\nNúmero de filas duplicadas en el dataset final: {num_duplicados}")

    # Dimensiones del dataset
    print("Dimensiones del dataset de clientes:")
    print(f'El dataset tiene {df_final.shape[0]} filas y {df_final.shape[1]} columnas.')
    
    # EXTRA: Generar un reporte de  las transformaciones aplicadas, por ejemplo:
    # Cantidad de filas iniciales y finales, cantidad de valores nulos por columna, cantidad de duplicados eliminados, etc.
    # Se puede exportar a un JSON o CSV o TEXT que sirva como registro de las transformaciones aplicadas a los datos.

if __name__ == "__main__":
    process_data()
