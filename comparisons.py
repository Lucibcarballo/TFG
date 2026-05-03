import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

from evaluation_graphics import preparar_datos


def generate_radar_comparative():
    # rutas tabla y encuesta
    ruta_csv_obj = "C:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_global.csv"
    ruta_excel_subj = (
        r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_notas.xlsx"
    )

    diccionario_metricas = {  # sacar de metricas como vayamos comparando y viendo similares
        # subjetivo : objetivo
        # "Sustain": "Sus(s)",
        "Brillantez": "Brillo (Global)",
        "Proyección": "Loud",
        # "Cuerpo": "Loud",
        "Claridad": "Sharp",
        # "Claridad": "Rough",
        "Equilibrio": "L/M (Global)",
    }

    # AJUSTAR DEPENDIENDO DE DATOS
    diccionario_audios = {
        1: "g5-ambos",  # Audio 1 = Nombre en el CSV
        2: "g4-ambos",  # Audio 2 = Nombre en el CSV
        3: "g3-ambos",
        4: "g2-ambos",
        5: "g1-ambos",
    }
    # =====================================================================

    # --- PREPARAR DATOS SUBJETIVOS (Encuesta) ---
    print("Procesando datos subjetivos...")
    df_subj, es_ranking = preparar_datos(
        ruta_excel_subj
    )  # funcion importada de evaluation_graphics.py que devuelve el dataframe y si es ranking o puntuacion

    # Filtramos solo los parámetros que hemos mapeado
    df_subj = df_subj[df_subj["Parametro"].isin(diccionario_metricas.keys())]

    # Calculamos la media de puntuación por Audio y Parámetro
    medias_subj = (
        df_subj.groupby(["Audio_Num", "Parametro"])["Puntuacion"].mean().reset_index()
    )

    # Normalizamos (0 a 1).
    # Si es ranking (1 a 5), el 1 es lo mejor (1.0) y el 5 lo peor (0.0).
    # Si es puntuación (0 a 10), dividimos entre 10.
    if es_ranking:
        medias_subj["Valor_Norm"] = 1 - ((medias_subj["Puntuacion"] - 1) / 4)
    else:
        medias_subj["Valor_Norm"] = medias_subj["Puntuacion"] / 10.0

    # --- PREPARAR DATOS OBJETIVOS (CSV) ---
    print("Procesando datos objetivos...")
    df_obj = pd.read_csv(ruta_csv_obj)

    columnas_obj_necesarias = list(diccionario_metricas.values())
    medias_obj = df_obj.groupby("Archivo")[columnas_obj_necesarias].mean().reset_index()

    # Normalizamos (Min-Max) los datos objetivos de 0 a 1
    for col in columnas_obj_necesarias:
        max_val = df_obj[col].max()
        min_val = df_obj[col].min()
        rango = max_val - min_val
        if rango != 0:
            medias_obj[col] = (medias_obj[col] - min_val) / rango
        else:
            medias_obj[col] = 0.0

    # --- RADAR CHART POR CADA AUDIO ---
    print("Generando Radar Charts superpuestos...")

    categorias_subj = list(diccionario_metricas.keys())
    categorias_obj = [diccionario_metricas[c] for c in categorias_subj]
    N = len(categorias_subj)

    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]

    for num_audio, nombre_csv in diccionario_audios.items():
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Eje X (Nombres de la encuesta)
        plt.xticks(
            angulos[:-1], categorias_subj, color="black", size=11, fontweight="bold"
        )

        # Eje Y (0 a 1)
        ax.set_rlabel_position(0)
        plt.yticks(
            [0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=10
        )
        plt.ylim(0, 1.1)

        # Extraer y dibujar datos SUBJETIVOS (Encuesta)
        datos_subj_audio = medias_subj[medias_subj["Audio_Num"] == num_audio]
        if not datos_subj_audio.empty:
            # Aseguramos el orden correcto
            datos_subj_audio = datos_subj_audio.set_index("Parametro").reindex(
                categorias_subj
            )
            valores_subj = datos_subj_audio["Valor_Norm"].values.flatten().tolist()
            valores_subj += valores_subj[:1]

            ax.plot(
                angulos,
                valores_subj,
                linewidth=2,
                linestyle="dashed",
                label="Percepción (Oyentes)",
                color="#e74c3c",
            )
            ax.fill(angulos, valores_subj, color="#e74c3c", alpha=0.1)

        # Extraer y dibujar datos OBJETIVOS (Código)
        datos_obj_audio = medias_obj[medias_obj["Archivo"] == nombre_csv]
        if not datos_obj_audio.empty:
            valores_obj = datos_obj_audio[categorias_obj].values.flatten().tolist()
            valores_obj += valores_obj[:1]

            ax.plot(
                angulos,
                valores_obj,
                linewidth=2.5,
                linestyle="solid",
                label="Análisis Acústico (Script)",
                color="#4195cc",
            )
            ax.fill(angulos, valores_obj, color="#4195cc", alpha=0.25)

        # Retoques
        plt.title(
            f"Audio {num_audio} ({nombre_csv}): Análisis vs Percepción",
            size=15,
            y=1.08,
            fontweight="bold",
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        nombre_salida = f"radar_comparativo_audio_{num_audio}.png"
        plt.savefig(nombre_salida, dpi=300, bbox_inches="tight")
        print(f" -> Guardado: {nombre_salida}")
        plt.close()


def generate_points_comparative():
    import seaborn as sns
    import matplotlib.lines as mlines

    # Rutas (las mismas que en el radar)
    ruta_csv_obj = "C:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_global.csv"
    ruta_excel_subj = (
        r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_notas.xlsx"
    )

    # Diccionarios (ajusta según tus métricas, sin claves repetidas)
    diccionario_metricas = {
        "Brillantez": "Brillo (Global)",
        "Proyección": "Loud",
        # "Sustain": "Sus(s)",
        "Cuerpo": "Loud",
        "Claridad": "Sharp",
        "Equilibrio": "L/M (Global)",
    }

    diccionario_audios = {
        1: "g5-ambos",
        2: "g4-ambos",
        3: "g3-ambos",
        4: "g2-ambos",
        5: "g1-ambos",
    }

    # 1. PREPARAR DATOS SUBJETIVOS (Oyentes)
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)
    df_subj = df_subj[df_subj["Parametro"].isin(diccionario_metricas.keys())]

    # 2. PREPARAR DATOS OBJETIVOS (Script)
    df_obj = pd.read_csv(ruta_csv_obj)
    columnas_obj = list(set(diccionario_metricas.values()))

    # Agrupamos por Archivo
    medias_obj = df_obj.groupby("Archivo")[columnas_obj].mean().reset_index()

    # Asignamos el Audio_Num correspondiente (1, 2, 3...) invirtiendo diccionario_audios
    map_inv_audios = {v: k for k, v in diccionario_audios.items()}
    medias_obj["Audio_Num"] = medias_obj["Archivo"].map(map_inv_audios)
    medias_obj = medias_obj.dropna(subset=["Audio_Num"]).sort_values("Audio_Num")

    # Normalizamos el dato objetivo para que encaje en el eje Y de la gráfica
    for col in columnas_obj:
        max_val = medias_obj[col].max()
        min_val = medias_obj[col].min()
        rango = max_val - min_val

        # Normalizado 0 a 1
        norm_01 = (medias_obj[col] - min_val) / rango if rango != 0 else 0.5

        if es_ranking:
            # En ranking: 1 es lo mejor (arriba) y 5 lo peor (abajo)
            # Valor objetivo más alto -> Puesto 1. Valor más bajo -> Puesto 5.
            medias_obj[col + "_escalado"] = 5 - (4 * norm_01)
        else:
            # En puntuación: Va de 0 a 10
            medias_obj[col + "_escalado"] = norm_01 * 10

    # 3. CREAR LA GRÁFICA BASE (Catplot)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    g = sns.catplot(
        data=df_subj,
        x="Audio_Num",
        y="Puntuacion",
        hue="Nombre",
        col="Parametro",
        col_wrap=2,
        kind="swarm",
        s=10,
        linewidth=1,
        alpha=0.7,
        height=4.5,
        aspect=1.5,
        palette="Set1",
    )

    if es_ranking:
        g.set(yticks=range(1, 6), ylim=(5.5, 0.5))
        y_label = "Puesto Ranking"
    else:
        g.set(yticks=range(0, 11))
        y_label = "Puntuación (0-10)"

    g.set_axis_labels("Número de Audio", y_label)
    g.set_titles("{col_name}", size=16, weight="bold")

    # 4. DIBUJAR LOS DATOS OBJETIVOS ENCIMA
    audios_unicos = sorted(df_subj["Audio_Num"].dropna().unique())

    for ax in g.axes.flat:
        ax.set_xticks(range(len(audios_unicos)))
        ax.set_xticklabels([f"Audio {int(i)}" for i in audios_unicos])

        # El título de cada subplot es el Parámetro (ej. "Brillantez")
        parametro_subj = ax.get_title()

        if parametro_subj in diccionario_metricas:
            col_obj = diccionario_metricas[parametro_subj]

            # Coordenadas: Seaborn posiciona las categorías en X=0, 1, 2, 3...
            x_coords = medias_obj["Audio_Num"] - 1
            y_coords = medias_obj[col_obj + "_escalado"]

            # Dibujamos una línea discontinua con una estrella para el dato objetivo
            ax.plot(
                x_coords,
                y_coords,
                color="black",
                marker="*",
                markersize=18,
                linestyle="dashed",
                linewidth=2.5,
                zorder=10,
            )

    # Añadir leyenda extra para explicar qué es la estrella negra
    estrella = mlines.Line2D(
        [],
        [],
        color="black",
        marker="*",
        linestyle="dashed",
        markersize=14,
        label="Análisis Acústico (Normalizado)",
    )
    g.fig.legend(handles=[estrella], loc="lower center", bbox_to_anchor=(0.5, -0.05))

    g.figure.suptitle(
        "Evaluación Subjetiva vs Análisis Objetivo", y=1.05, fontsize=18, weight="bold"
    )

    plt.savefig("grafico_mixto_overlay.png", dpi=300, bbox_inches="tight")
    print("Guardado: grafico_mixto_overlay.png")
    plt.close()


if __name__ == "__main__":
    generate_radar_comparative()
    generate_points_comparative()
