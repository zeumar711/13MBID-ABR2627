# Importación de librerías y supresión de advertencias
from dotenv import variables
import matplotlib
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sweetviz as sv




warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)

def visualize_data(datos_clientes: str= "data/raw/datos_clientes.csv",
                   datos_resenas: str= "data/raw/datos_resenas.csv",
                   output_dir: str= "docs/figures/") -> None:
    """
    Función para visualizar los datos de clientes y reseñas.
    """
    # Creación de directorio de salida si no existe
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lectura de los datos
    df_clientes = pd.read_csv(datos_clientes, sep=";")
    df_resenas = pd.read_csv(datos_resenas, sep=";")

    # Configuración general
    sns.set(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (8, 5)

    # Gráfico 1: Distribución de los valores de la variable target
    orden_valores_target = ["Negativo", "Neutral", "Positivo"]
    sns.countplot(x="sentimiento", data=df_resenas, order=orden_valores_target)
    plt.title("Distribución de la variable objetivo (sentimiento)")
    plt.xlabel("Sentimiento expresado en la reseña")
    plt.ylabel("Cantidad de reseñas")
    plt.savefig(output_dir / "distribucion_sentimiento.png")
    plt.close()

    df_resenas["sentimiento"].value_counts(normalize=True).mul(100).round(2)

    # Gráfico 2: Longitud del texto de las reseñas
    df_resenas["longitud_texto"] = df_resenas["texto_resena"].str.len()

    sns.boxplot(x="sentimiento", y="longitud_texto", data=df_resenas, order=["Negativo", "Neutral", "Positivo"])
    plt.title("Longitud del texto de la reseña según sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad de caracteres")
    plt.savefig(output_dir / "longitud_texto.png")
    plt.close()

    # se descarta la columna auxiliar, ya que solo se utilizó para esta visualización
    df_resenas.drop(columns=["longitud_texto"], inplace=True)

    # Gráfico 3: Matriz de correlación entre variables numéricas_Clientes    
    num_df = df_clientes.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Clientes')
    plt.savefig(output_dir / "matriz_correlacion_clientes.png")
    plt.close()

    # Gráfico 4: Matriz de correlación entre variables numéricas_Reseñas
    num_df = df_resenas.select_dtypes(include=['float64', 'int64'])
    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones - Reseñas')
    plt.savefig(output_dir / "matriz_correlacion_resenas.png")
    plt.close()


####
#TODO: Agregar por lo menos nos visualización más del escenario
# Posiblemente alguna que vincule a alguno de los atributos con la variable objetivo.
# Por ejemplo: sentimiento según tiempo de respuesta.
# También puede ser la distribución de valores de otras variables.


    # Gráfico 5: Sentimiento de acuerdo con la calificación.
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df_resenas,
        x="sentimiento",
        y="calificacion",
        order=["Negativo", "Neutral", "Positivo"]
    )

    plt.title("Calificación según sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Calificación")
    plt.savefig(output_dir / "calificacion_sentimiento.png")
    plt.close()

    # Gráfico 6: Porcentaje por categoría de producto y sentimiento.
    tabla_pct = pd.crosstab(
        df_resenas['categoria_producto'],
        df_resenas['sentimiento'],
        normalize='index'
    ) * 100
    print(tabla_pct.shape)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        tabla_pct,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        linewidths=0.5
    )

    plt.title('Porcentaje de sentimiento por categoría de producto')
    plt.xlabel('Sentimiento')
    plt.ylabel('Categoría de producto')

    # Añadir símbolo %
    for t in plt.gca().texts:
        t.set_text(t.get_text() + '%')

    plt.savefig(output_dir / "Mapa_sentimiento_por_categoria_producto.png")
    plt.close()

    # Gráfico 7: Distribución de reseñas por categoría de producto
    plt.figure(figsize=(10, 6))
    (
        df_resenas["categoria_producto"]
        .value_counts()
        .sort_values()
        .plot(kind="barh", color="steelblue")
    )

    plt.title("Distribución de reseñas por categoría de producto")
    plt.xlabel("Número de reseñas")
    plt.ylabel("Categoría de producto")

    plt.tight_layout()
    plt.savefig(output_dir / "distribucion_categoria_producto.png")
    plt.close()

#EXTRA: Generar un reporte automático con alguna librería como sweetviz, dtale.

    # =====================================================
    # REPORTE AUTOMÁTICO SWEETVIZ
    # + GRÁFICO TIEMPO DE RESPUESTA VS SENTIMIENTO
    # =====================================================

    # -----------------------------------------------------
    # Directorio de salida
    # -----------------------------------------------------
    matplotlib.rcParams["font.family"] = "Arial"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Preparación de datos para Sweetviz
    # -----------------------------------------------------

    df_sw = df_resenas.copy()

    # Agrupar tiempos de respuesta

    df_sw["rango_respuesta"] = pd.cut(
        df_sw["tiempo_respuesta_horas"],
        bins=[0, 6, 12, 24, 48, float("inf")],
        labels=[
            "0-6 h",
            "6-12 h",
            "12-24 h",
            "24-48 h",
            ">48 h"
        ]
    )

    # Agrupar importe de compra

    df_sw["rango_importe"] = pd.qcut(
        df_sw["importe_compra"],
        q=4,
        duplicates="drop"
    )

    # -----------------------------------------------------
    # REPORTE SWEETVIZ
    # -----------------------------------------------------

    report = sv.analyze(df_sw)

    report.show_html(
        filepath=str(output_dir / "reporte_sentimiento_sweetviz.html"),
        open_browser=True
    )

    print("✅ Reporte Sweetviz generado correctamente")

    # -----------------------------------------------------
    # GRÁFICO 1:
    # Tiempo de respuesta según sentimiento
    # -----------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df_resenas,
        x="sentimiento",
        y="tiempo_respuesta_horas",
        order=["Negativo", "Neutral", "Positivo"]
    )

    plt.title("Tiempo de respuesta según sentimiento")
    plt.xlabel("Sentimiento")
    plt.ylabel("Tiempo de respuesta (horas)")

    plt.tight_layout()

    plt.savefig(
        output_dir / "tiempo_respuesta_sentimiento.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Gráfico tiempo_respuesta_sentimiento.png generado")

    # -----------------------------------------------------
    # GRÁFICO 2:
    # Distribución del sentimiento por rangos de respuesta
    # -----------------------------------------------------

    tabla_sentimiento = pd.crosstab(
        df_sw["rango_respuesta"],
        df_sw["sentimiento"],
        normalize="index"
    ) * 100

    # Ordenar columnas si existen
    columnas = [
        c for c in ["Negativo", "Neutral", "Positivo"]
        if c in tabla_sentimiento.columns
    ]

    tabla_sentimiento = tabla_sentimiento[columnas]

    ax = tabla_sentimiento.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 6),
        color=["#d73027", "#fee08b", "#1a9850"][:len(columnas)]
    )

    plt.title(
        "Distribución del sentimiento según tiempo de respuesta"
    )

    plt.xlabel("Porcentaje (%)")
    plt.ylabel("Rango de tiempo de respuesta")

    plt.legend(
        title="Sentimiento",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "sentimiento_por_tiempo_respuesta.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Gráfico sentimiento_por_tiempo_respuesta.png generado")
    
####

if __name__ == "__main__":
    visualize_data()