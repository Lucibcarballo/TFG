import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generar_multiples_graficas_puntos_visibles(ruta_excel):
    df = pd.read_excel(ruta_excel)

    id_vars = ["Hora", "Nombre", "Comentario"]
    columnas_fijas = [col for col in id_vars if col in df.columns]

    melted = df.melt(
        id_vars=columnas_fijas, var_name="Param_Audio", value_name="Puntuacion"
    )
    melted["Puntuacion"] = pd.to_numeric(melted["Puntuacion"], errors="coerce")
    melted = melted.dropna(subset=["Puntuacion"])

    melted["Param_Audio"] = melted["Param_Audio"].str.replace(
        "Sustain-", "Sustain -", regex=False
    )
    melted[["Parametro", "Audio"]] = melted["Param_Audio"].str.split(
        "-", n=1, expand=True
    )
    melted["Parametro"] = melted["Parametro"].str.strip()
    melted["Audio"] = melted["Audio"].str.strip()
    melted["Audio_Num"] = melted["Audio"].str.extract(r"(\d+)")[0].astype(int)

    sns.set_theme(style="whitegrid", font_scale=1.2)

    g = sns.catplot(
        data=melted,
        x="Audio_Num",
        y="Puntuacion",
        hue="Nombre",
        col="Parametro",
        col_wrap=2,
        kind="swarm",
        s=22,  # Tamaño de los puntos
        linewidth=1.5,  # Grosor del borde
        alpha=0.9,  # Colores más sólidos
        height=4.5,
        aspect=1.5,
        palette="Set1",
    )

    g.set_axis_labels("Número de Audio", "Puntuación (0-10)")
    g.set_titles("{col_name}", size=16, weight="bold")
    g.set(yticks=range(0, 11))

    for ax in g.axes.flat:
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"Aud {i}" for i in range(1, 11)])

    g.fig.suptitle(
        "Evaluación de cada Oyente separada por Parámetro",
        y=1.03,
        fontsize=18,
        weight="bold",
    )

    archivo_salida = "grafica_puntos_destacados.png"
    plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
    print(f"¡Gráfica guardada como {archivo_salida}!")
    plt.show()


if __name__ == "__main__":
    RUTA_ARCHIVO = r"c:\Users\lucib\Desktop\TFG\Encuesta.xlsx"
    generar_multiples_graficas_puntos_visibles(RUTA_ARCHIVO)
