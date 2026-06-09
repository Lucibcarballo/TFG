import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import seaborn as sns
import matplotlib.lines as mlines

from evaluation_graphics import preparar_datos

plt.rcParams.update(
    {
        "font.size": 18,  # Tamaño general
        "axes.titlesize": 22,  # Títulos de los sub-gráficos
        "axes.labelsize": 18,  # Etiquetas de los ejes
        "xtick.labelsize": 18,  # Letras del radar (exterior) y eje X
        "ytick.labelsize": 18,  # Porcentajes del radar y eje Y
        "legend.fontsize": 20,  # Leyenda
        "figure.titlesize": 24,  # Título superior general
    }
)


def generate_radar_comparative(
    ruta_csv_obj, ruta_excel_subj, diccionario_metricas, diccionario_audios
):
    # --- PREPARAR DATOS SUBJETIVOS (Encuesta) ---
    print("Procesando datos subjetivos para Radar Chart...")
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

    print("Audios encontrados en el Excel:", df_subj["Audio_Num"].unique())

    # Filtramos solo los parámetros que hemos mapeado y forzamos copia limpia
    df_subj = df_subj[df_subj["Parametro"].isin(diccionario_metricas.keys())].copy()

    # Normalizar e invertir FILA por FILA primero
    if es_ranking:
        print("Los datos subjetivos son RANKING. Invirtiendo a nivel de fila...")
        max_rank = df_subj["Puntuacion"].max()
        denom = (max_rank - 1) if max_rank > 1 else 1
        df_subj["Valor_Norm"] = 1.0 - ((df_subj["Puntuacion"] - 1.0) / denom)
    else:
        df_subj["Valor_Norm"] = df_subj["Puntuacion"] / 10.0
        
    # Ahora sí, agrupamos de forma única extrayendo la media del Valor_Norm corregido
    medias_subj = (
        df_subj.groupby(["Audio_Num", "Parametro"])["Valor_Norm"].mean().reset_index()
    )

    # --- PREPARAR DATOS OBJETIVOS (CSV) ---
    print("Procesando datos objetivos para Radar Chart...")
    df_obj = pd.read_csv(ruta_csv_obj)
    df_obj["Archivo"] = df_obj["Archivo"].str.strip()

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

    etiquetas_radar = [
        f"{subj}\nvs {obj}" for subj, obj in zip(categorias_subj, categorias_obj)
    ]

    N = len(categorias_subj)
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]

    for num_audio, nombre_csv in diccionario_audios.items():
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angulos[:-1], etiquetas_radar, color="black")
        ax.tick_params(axis="x", pad=35)

        ax.set_rlabel_position(0)
        plt.yticks(
            [0.25, 0.5, 0.75, 1.0],
            ["25%", "50%", "75%", "100%"],
            color="grey",
            fontsize=16,
        )
        plt.ylim(0, 1.1)

        datos_subj_audio = medias_subj[medias_subj["Audio_Num"] == num_audio]
        if not datos_subj_audio.empty:
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

        plt.title(
            f"Audio {num_audio} ({nombre_csv}): Objetivo vs Subjetivo",
            y=1.12,
            fontweight="bold",
        )

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            ncol=2,
        )

        nombre_salida = f"radar_comparativo_audio_{num_audio}.png"
        plt.savefig(nombre_salida, dpi=300, bbox_inches="tight")
        print(f" -> Guardado: {nombre_salida}")
        plt.close()


def generate_points_comparative(
    ruta_csv_obj, ruta_excel_subj, diccionario_metricas, diccionario_audios
):
    print("Procesando datos subjetivos para Gráfica de Puntos...")
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

    print("Procesando datos objetivos para Gráfica de Puntos...")
    df_obj = pd.read_csv(ruta_csv_obj)
    df_obj["Archivo"] = df_obj["Archivo"].str.strip()
    columnas_obj = list(set(diccionario_metricas.values()))

    medias_obj = df_obj.groupby("Archivo")[columnas_obj].mean().reset_index()

    map_inv_audios = {v: k for k, v in diccionario_audios.items()}
    medias_obj["Audio_Num"] = medias_obj["Archivo"].map(map_inv_audios)
    medias_obj = medias_obj.dropna(subset=["Audio_Num"]).sort_values("Audio_Num")

    for col in columnas_obj:
        if es_ranking:
            medias_obj[col + "_escalado"] = medias_obj[col].rank(
                ascending=False, method="min"
            )
        else:
            max_val = medias_obj[col].max()
            min_val = medias_obj[col].min()
            rango = max_val - min_val
            norm_01 = (medias_obj[col] - min_val) / rango if rango != 0 else 0.5
            medias_obj[col + "_escalado"] = norm_01 * 10

    print("Generando Gráfica de Puntos superpuesta...")
    sns.set_theme(style="whitegrid", font_scale=1.2)
    g = sns.catplot(
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

        if parametro_subj in diccionario_metricas:
            col_obj = diccionario_metricas[parametro_subj]
            x_coords = medias_obj["Audio_Num"] - 1
            y_coords = medias_obj[col_obj + "_escalado"]

            ax.plot(
                x_coords,
                y_coords,
                color="red",
                marker="s",
                markersize=8,
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.5,
                zorder=10,
            )
            ax.set_title(f"{parametro_subj}\nvs {col_obj}", weight="bold")
        else:
            ax.set_title(f"{parametro_subj}", weight="bold")

    cuadrado = mlines.Line2D(
        [],
        [],
        color="none",
        marker="s",
        linestyle="",
        markersize=10,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=1.5,
        label="Análisis objetivo",
    )
    g.figure.legend(handles=[cuadrado], loc="lower center", bbox_to_anchor=(0.5, -0.05))
    g.figure.suptitle("Comparativa objetivo-subjetiva", y=1.05, weight="bold")

    plt.savefig("grafico_mixto_overlay.png", dpi=300, bbox_inches="tight")
    print("Guardado: grafico_mixto_overlay.png")
    plt.close()


def generate_radar_guitarras_comparative(
    ruta_csv_obj, ruta_excel_subj, diccionario_metricas, mapeo_audio_guitarra
):
    print("Procesando datos subjetivos por Guitarra...")
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

    # Forzamos una copia explícita para evitar warnings de Pandas al modificar columnas
    df_subj = df_subj[df_subj["Parametro"].isin(diccionario_metricas.keys())].copy()
    df_subj["Guitarra"] = df_subj["Audio_Num"].map(mapeo_audio_guitarra)
    df_subj = df_subj.dropna(subset=["Guitarra"]).copy()

    # Normalización e inversión individual limpia antes del agrupamiento definitivo
    if es_ranking:
        max_rank_subj = df_subj["Puntuacion"].max()
        denom_subj = (max_rank_subj - 1) if max_rank_subj > 1 else 1
        df_subj["Valor_Norm"] = 1.0 - ((df_subj["Puntuacion"] - 1.0) / denom_subj)
    else:
        df_subj["Valor_Norm"] = df_subj["Puntuacion"] / 10.0
        
    # Ahora creamos el medias_subj definitivo directamente sobre la columna ya tratada
    medias_subj = (
        df_subj.groupby(["Guitarra", "Parametro"])["Valor_Norm"].mean().reset_index()
    )

    print("Procesando datos objetivos por Guitarra convirtiendo a RANKING...")
    df_obj = pd.read_csv(ruta_csv_obj)
    df_obj["Guitarra"] = df_obj["Guitarra"].str.strip()

    columnas_obj_necesarias = list(set(diccionario_metricas.values()))
    medias_obj = df_obj.groupby("Guitarra")[columnas_obj_necesarias].mean().reset_index()

    num_guitarras_obj = len(medias_obj)
    denom_obj = (num_guitarras_obj - 1) if num_guitarras_obj > 1 else 1

    for col in columnas_obj_necesarias:
        puestos_fisicos = medias_obj[col].rank(ascending=False, method="min")
        medias_obj[col] = 1 - ((puestos_fisicos - 1) / denom_obj)

    print("Generando Radar Charts comparativos de Guitarras...")

    categorias_subj = list(diccionario_metricas.keys())
    categorias_obj = [diccionario_metricas[c] for c in categorias_subj]
    etiquetas_radar = [
        f"{subj}\nvs {obj}" for subj, obj in zip(categorias_subj, categorias_obj)
    ]

    N = len(categorias_subj)
    angulos = [n / float(N) * 2 * pi for n in range(N)]
    angulos += angulos[:1]

    guitarras_unicas = medias_obj["Guitarra"].unique()

    for guitarra in guitarras_unicas:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angulos[:-1], etiquetas_radar, color="black")
        ax.tick_params(axis="x", pad=15)

        ax.set_rlabel_position(0)
        plt.yticks(
            [0.25, 0.5, 0.75, 1.0],
            ["25%", "50%", "75%", "100%"],
            color="grey",
            fontsize=16,
        )
        plt.ylim(0, 1.1)

        datos_subj_guit = medias_subj[medias_subj["Guitarra"] == guitarra]
        if not datos_subj_guit.empty:
            datos_subj_guit = datos_subj_guit.set_index("Parametro").reindex(
                categorias_subj
            )
            valores_subj = datos_subj_guit["Valor_Norm"].values.flatten().tolist()
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

        datos_obj_guit = medias_obj[medias_obj["Guitarra"] == guitarra]
        if not datos_obj_guit.empty:
            valores_obj = datos_obj_guit[categorias_obj].values.flatten().tolist()
            valores_obj += valores_obj[:1]

            ax.plot(
                angulos,
                valores_obj,
                linewidth=2.5,
                linestyle="solid",
                label="Análisis objetivo (Promedio Notas)",
                color="#2ecc71",
            )
            ax.fill(angulos, valores_obj, color="#2ecc71", alpha=0.25)

            num_guitarra = guitarra.lower().replace("g", "")

            plt.title(
                f"Guitarra {num_guitarra}: Comparativa objetivo-subjetiva",
                y=1.15,
                fontweight="bold",
            )

            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 0),
                ncol=1,
            )

            nombre_salida = f"radar_comparativo_guitarra_{num_guitarra}.png"
            plt.savefig(nombre_salida, dpi=300, bbox_inches="tight")
            print(f" -> Guardado: {nombre_salida}")
            plt.close()


if __name__ == "__main__":
    RUTA_CSV_OBJ = "c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos_con_calibracion\\dataset_guitarras_grabaciones_global.csv"
    RUTA_CSV_PROMEDIOS_NOTAS = "c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos_con_calibracion\\dataset_guitarras_promedio_notas.csv"
    RUTA_EXCEL_SUBJ = r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_notas.xlsx"
    
    DICCIONARIO_METRICAS = {
        "Brillantez": "Brillo (Global)",
        "Proyección": "Loud",
        "Cuerpo": "Loud",
        "Claridad": "Sharp",
        "Equilibrio": "L/M (Global)",
    }

    DICCIONARIO_METRICAS_PROMEDIOS_NOTAS = {
        "Brillantez": "Brillo (Nota)",
        "Sustain": "Sus(s)",
        "Equilibrio": "L/M (Nota)",
    }

    DICCIONARIO_AUDIOS = {
        1: "g5-ambos",
        2: "g4-ambos",
        3: "g3-ambos",
        4: "g2-ambos",
        5: "g1-ambos",
    }

    MAPEO_AUDIO_GUITARRA = {   # se hace el mapeo inverso en evaluation_graphics.py
        1: "g1",                    
        2: "g2",
        3: "g3",
        4: "g4",
        5: "g5",
    }

    generate_radar_guitarras_comparative(
        RUTA_CSV_PROMEDIOS_NOTAS,
        RUTA_EXCEL_SUBJ,
        DICCIONARIO_METRICAS_PROMEDIOS_NOTAS,
        MAPEO_AUDIO_GUITARRA,
    )

    generate_radar_comparative(
        RUTA_CSV_OBJ, RUTA_EXCEL_SUBJ, DICCIONARIO_METRICAS, DICCIONARIO_AUDIOS
    )

    print("-" * 30)

    generate_points_comparative(
        RUTA_CSV_OBJ, RUTA_EXCEL_SUBJ, DICCIONARIO_METRICAS, DICCIONARIO_AUDIOS
    )

    print("--- PROCESO FINALIZADO ---")