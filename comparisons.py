import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import seaborn as sns
import matplotlib.lines as mlines

from evaluation_graphics import preparar_datos


def generate_radar_comparative(
    ruta_csv_obj, ruta_excel_subj, diccionario_metricas, diccionario_audios
):
    # --- PREPARAR DATOS SUBJETIVOS (Encuesta) ---
    print("Procesando datos subjetivos para Radar Chart...")
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

    print("Audios encontrados en el Excel:", df_subj["Audio_Num"].unique())

    # Filtramos solo los parámetros que hemos mapeado
    df_subj = df_subj[df_subj["Parametro"].isin(diccionario_metricas.keys())]

    # Calculamos la media de puntuación por Audio y Parámetro
    medias_subj = (
        df_subj.groupby(["Audio_Num", "Parametro"])["Puntuacion"].mean().reset_index()
    )

    # Normalizamos (0 a 1).
    if es_ranking:
        print("Los datos subjetivos son RANKING. ")
        medias_subj["Valor_Norm"] = 1 - ((medias_subj["Puntuacion"] - 1) / 4)
    else:
        medias_subj["Valor_Norm"] = medias_subj["Puntuacion"] / 10.0

    # --- PREPARAR DATOS OBJETIVOS (CSV) ---
    print("Procesando datos objetivos para Radar Chart...")
    df_obj = pd.read_csv(ruta_csv_obj)
    df_obj["Archivo"] = df_obj["Archivo"].str.strip()

    # Usamos set() para evitar columnas duplicadas al agrupar
    columnas_obj_necesarias = list(set(diccionario_metricas.values()))
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

    # --- NUEVO: Etiquetas combinadas para el gráfico ---
    etiquetas_radar = [
        f"{subj}\nvs {obj}" for subj, obj in zip(categorias_subj, categorias_obj)
    ]

    N = len(categorias_subj)

    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]

    for num_audio, nombre_csv in diccionario_audios.items():
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Eje X (Nombres de la encuesta + Nombre Objetivo)
        plt.xticks(
            angulos[:-1],
            etiquetas_radar,
            color="black",
            size=11,
            fontweight="bold",
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
                label="Subjetivo (Encuesta músicos)",
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
                label="Análisis objetivo (Código)",
                color="#4195cc",
            )
            ax.fill(angulos, valores_obj, color="#4195cc", alpha=0.25)

        # Retoques
        plt.title(
            f"Audio {num_audio} ({nombre_csv}): Objetivo vs Subjetivo",
            size=15,
            y=1.08,
            fontweight="bold",
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        nombre_salida = f"radar_comparativo_audio_{num_audio}.png"
        plt.savefig(nombre_salida, dpi=300, bbox_inches="tight")
        print(f" -> Guardado: {nombre_salida}")
        plt.close()


def generate_points_comparative(
    ruta_csv_obj, ruta_excel_subj, diccionario_metricas, diccionario_audios
):
    # 1. PREPARAR DATOS SUBJETIVOS (Oyentes)
    print("Procesando datos subjetivos para Gráfica de Puntos...")
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

    # 2. PREPARAR DATOS OBJETIVOS (Script)
    print("Procesando datos objetivos para Gráfica de Puntos...")
    df_obj = pd.read_csv(ruta_csv_obj)
    df_obj["Archivo"] = df_obj["Archivo"].str.strip()
    columnas_obj = list(set(diccionario_metricas.values()))

    # Agrupamos por Archivo
    medias_obj = df_obj.groupby("Archivo")[columnas_obj].mean().reset_index()

    # Asignamos el Audio_Num correspondiente (1, 2, 3...) invirtiendo diccionario_audios
    map_inv_audios = {v: k for k, v in diccionario_audios.items()}
    medias_obj["Audio_Num"] = medias_obj["Archivo"].map(map_inv_audios)
    medias_obj = medias_obj.dropna(subset=["Audio_Num"]).sort_values("Audio_Num")

    # Normalizamos el dato objetivo para que encaje en el eje Y de la gráfica
    # Convertimos el dato objetivo en Ranking o lo Normalizamos
    for col in columnas_obj:
        if es_ranking:
            # Convertimos los valores acústicos en posiciones (1º, 2º, 3º, 4º, 5º)
            # ascending=False significa que el valor acústico MÁS ALTO se lleva el PUESTO 1.
            # method='min' hace que si dos audios tienen EXACTAMENTE el mismo valor, compartan puesto.
            medias_obj[col + "_escalado"] = medias_obj[col].rank(
                ascending=False, method="min"
            )
        else:
            # Si en la encuesta en vez de ranking disteis puntuaciones (0 a 10),
            # normalizamos proporcionalmente como hacíamos antes:
            max_val = medias_obj[col].max()
            min_val = medias_obj[col].min()
            rango = max_val - min_val

            norm_01 = (medias_obj[col] - min_val) / rango if rango != 0 else 0.5
            medias_obj[col + "_escalado"] = norm_01 * 10

    print("Generando Gráfica de Puntos superpuesta...")
    sns.set_theme(style="whitegrid", font_scale=1.2)
    g = sns.catplot(  # puntos encuesta
        data=df_subj,
        x="Audio_Num",
        y="Puntuacion",
        hue="Nombre",
        col="Parametro",
        col_wrap=2,
        kind="swarm",
        s=15,
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

    audios_unicos = sorted(df_subj["Audio_Num"].dropna().unique())

    for parametro_subj, ax in zip(g.col_names, g.axes.flat):
        ax.set_xticks(range(len(audios_unicos)))
        ax.set_xticklabels([f"Audio {int(i)}" for i in audios_unicos])

        # Verificamos si este parámetro tiene una equivalencia acústica en el diccionario
        if parametro_subj in diccionario_metricas:
            col_obj = diccionario_metricas[parametro_subj]

            # Coordenadas: Seaborn posiciona las categorías en X=0, 1, 2, 3...
            x_coords = medias_obj["Audio_Num"] - 1
            y_coords = medias_obj[col_obj + "_escalado"]

            ax.plot(
                x_coords,
                y_coords,
                color="red",
                marker="s",  # s = square
                markersize=8,
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.5,
                zorder=10,
            )

            # Cambiamos el título para mostrar la comparativa exacta
            ax.set_title(f"{parametro_subj}\nvs {col_obj}", size=14, weight="bold")
        else:
            # Si no hay equivalencia objetiva (ej. Sustain), dejamos solo el nombre limpio
            ax.set_title(f"{parametro_subj}", size=14, weight="bold")

    cuadrado = mlines.Line2D(
        [],
        [],
        color="none",
        marker="s",
        linestyle="",
        markersize=10,
        markerfacecolor="none",  # INTERIOR TRANSPARENTE
        markeredgecolor="red",
        markeredgewidth=1.5,
        label="Análisis objetivo",
    )
    g.figure.legend(handles=[cuadrado], loc="lower center", bbox_to_anchor=(0.5, -0.05))

    g.figure.suptitle(
        "Evaluación Subjetiva vs Análisis Objetivo", y=1.05, fontsize=18, weight="bold"
    )

    plt.savefig("grafico_mixto_overlay.png", dpi=300, bbox_inches="tight")
    print("Guardado: grafico_mixto_overlay.png")
    plt.close()


if __name__ == "__main__":

    # ================= CONFIGURACIÓN GLOBAL =================

    # Rutas de los archivos de datos
    # RUTA_CSV_OBJ = "C:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_global.csv"
    RUTA_CSV_OBJ = "c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\piezas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_global.csv"

    RUTA_EXCEL_SUBJ = (
        r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_piezas.xlsx"
    )

    # Mapeo de Parámetro Subjetivo (Encuesta) -> Parámetro Objetivo (Código/CSV)
    DICCIONARIO_METRICAS = {
        "Brillantez": "Brillo (Global)",
        "Proyección": "Loud",
        "Cuerpo": "Loud",
        "Claridad": "Sharp",
        "Equilibrio": "L/M (Global)",
    }

    # Mapeo del Número de Audio de la encuesta -> Nombre del archivo acústico en el CSV
    # DICCIONARIO_AUDIOS = {
    #     1: "g5-ambos",
    #     2: "g4-ambos",
    #     3: "g3-ambos",
    #     4: "g2-ambos",
    #     5: "g1-ambos",
    # }

    DICCIONARIO_AUDIOS = {
        1: "guitarra 1-415Hzalejandro pieza",
        2: "guitarra 1-415Hzuxia pieza",
        3: "guitarra2-alejandro pieza",
        4: "guitarra2-uxia pieza",
        5: "guitarra3-alejandro pieza",
        6: "guitarra3-uxia pieza",
        7: "guitarra4-alejandro pieza",
        8: "guitarra4-uxia pieza",
    }

    # ================= EJECUCIÓN =================

    print("--- INICIANDO COMPARATIVA ---")

    # Pasamos las variables como argumentos a cada función
    generate_radar_comparative(
        RUTA_CSV_OBJ, RUTA_EXCEL_SUBJ, DICCIONARIO_METRICAS, DICCIONARIO_AUDIOS
    )

    print("-" * 30)

    generate_points_comparative(
        RUTA_CSV_OBJ, RUTA_EXCEL_SUBJ, DICCIONARIO_METRICAS, DICCIONARIO_AUDIOS
    )

    print("--- PROCESO FINALIZADO ---")
