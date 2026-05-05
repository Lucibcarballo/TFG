import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preparar_datos(ruta_excel):
    df = pd.read_excel(ruta_excel)

    id_vars = ["Hora", "Nombre", "Comentario"]
    columnas_fijas = [col for col in id_vars if col in df.columns]

    # aplanamos el DataFrame para tener una fila por cada combinación de oyente-parámetro-puntuación
    melted = df.melt(id_vars=columnas_fijas, var_name="Columna", value_name="Valor")
    melted = melted.dropna(subset=["Valor"])

    # DETECTAMOS QUE TIPO DE ENCUESTA ES
    es_ranking = (
        melted["Valor"].astype(str).str.contains("Audio", case=False, na=False).any()
    )

    melted["Columna"] = melted["Columna"].str.replace(
        "Sustain-", "Sustain -", regex=False
    )
    melted[["Parametro", "Subcolumna"]] = melted["Columna"].str.split(
        "-", n=1, expand=True
    )
    melted["Parametro"] = melted["Parametro"].str.strip()
    melted["Subcolumna"] = melted["Subcolumna"].str.strip()

    if es_ranking:
        # En Ranking: "Audio X" Extraemos ese num X
        melted["Audio_Num"] = (
            melted["Valor"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        )

        # En Ranking: la puntuación (Puesto 1, 2, 3...) viene del nombre de la columna (ej: "Sustain - Puesto1")
        melted["Puntuacion"] = (
            melted["Subcolumna"].str.extract(r"Puesto\s*(\d+)")[0].astype(int)
        )

    else:
        # En Puntuacion: la celda tiene el valor numerico (0-10)
        melted["Puntuacion"] = pd.to_numeric(melted["Valor"], errors="coerce")
        melted = melted.dropna(subset=["Puntuacion"])

        # En Puntuacion: el num audio viene de la columna (ej: "Sustain - Audio 1")
        melted["Audio_Num"] = melted["Subcolumna"].str.extract(r"(\d+)")[0].astype(int)

    # # OJO MAPEAR SEGUN ORDEN AUDIOS ENCUESTA
    # mapeo_inverso = {5: 1, 4: 2, 3: 3, 2: 4, 1: 5}
    # melted["Audio_Num"] = melted["Audio_Num"].map(mapeo_inverso)

    # Ordenamos por numero de guitarra
    melted = melted.sort_values(by="Audio_Num")

    orden_parametros = [
        "Brillantez",
        "Proyección",
        "Sustain",
        "Cuerpo",
        "Claridad",
        "Equilibrio",
    ]
    melted["Parametro"] = pd.Categorical(
        melted["Parametro"], categories=orden_parametros, ordered=True
    )

    return melted, es_ranking


def generar_grafica_puntos(df, es_ranking):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    g = sns.catplot(
        data=df,
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

    if es_ranking:
        y_label = "Puesto Ranking"
        g.set(
            yticks=range(1, 6), ylim=(5.5, 0.5)
        )  # Invertimos el eje Y para que 1 esté arriba
    else:
        y_label = "Puntuación (0-10)"
        g.set(yticks=range(0, 11))

    g.set_axis_labels("Número de Audio", y_label)
    g.set_titles("{col_name}", size=16, weight="bold")

    audios_unicos = sorted(df["Audio_Num"].dropna().unique())
    for ax in g.axes.flat:
        ax.set_xticks(range(len(audios_unicos)))
        ax.set_xticklabels([f"Audio {int(i)}" for i in audios_unicos])

    titulo = "Evaluación por Ranking" if es_ranking else "Evaluación por Puntuación"
    g.figure.suptitle(
        f"{titulo} separada por parámetro", y=1.03, fontsize=18, weight="bold"
    )

    archivo_salida = (
        "grafica_puntos_ranking.png" if es_ranking else "grafica_puntos_puntuacion.png"
    )
    plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
    print(f"[OK]: Gráfica guardada como {archivo_salida}")
    plt.show()


def generar_boxplot_global(df, es_ranking):
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid", font_scale=1.1)

    sns.boxplot(
        data=df,
        x="Audio_Num",
        y="Puntuacion",
        hue="Parametro",
        palette="pastel",
        linewidth=1,  # Grosor de las líneas de la caja
        fliersize=4,  # Tamaño de los outliers
        boxprops=dict(alpha=0.9),
    )

    titulo = (
        "Distribución de Rankings" if es_ranking else "Distribución de Puntuaciones"
    )
    plt.title(f"{titulo} por Audio (todos los oyentes)", fontsize=18, pad=15)
    plt.xlabel("Número de Audio", labelpad=10, fontsize=12)

    if es_ranking:
        plt.ylabel("Posición (1 = Mejor, 5 = Peor)", labelpad=10, fontsize=12)
        plt.yticks(range(1, 6))
        plt.gca().invert_yaxis()  # Invertimos el eje Y
    else:
        plt.ylabel("Puntuación (0-10)", labelpad=10, fontsize=12)

    # Ajustamos las etiquetas del eje X para que diga "Audio X" de forma segura
    audios_unicos = sorted(df["Audio_Num"].dropna().unique())
    plt.xticks(
        ticks=range(len(audios_unicos)),
        labels=[f"Audio {int(i)}" for i in audios_unicos],
    )

    plt.legend(
        title="Parámetro", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True
    )
    plt.tight_layout()

    archivo_salida = "boxplot_ranking.png" if es_ranking else "boxplot_puntuacion.png"
    plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
    print(f"[OK]: Boxplot guardado como {archivo_salida}")
    plt.show()


if __name__ == "__main__":
    RUTA_ARCHIVO = (
        r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_notas.xlsx"
    )

    df_limpio, es_ranking = preparar_datos(RUTA_ARCHIVO)

    generar_grafica_puntos(df_limpio, es_ranking)
    generar_boxplot_global(df_limpio, es_ranking)
