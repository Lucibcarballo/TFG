import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os

from evaluation_graphics import preparar_datos


def generate_radar_comparative():
    # rutas tabla y encuesta
    ruta_csv_obj = "dataset_guitarras_grabaciones_notas.csv"
    ruta_excel_subj = (
        r"C:\Users\lucib\Desktop\TFG\RESULTADOS\encuestas\Encuesta_notas.xlsx"
    )

    diccionario_metricas = {  # sacar de metricas como vayamos comparando y viendo similares
        # subjetivo : objetivo
        "Sustain": "Sus(s)",
        "Brillantez": "Brillo (Global)",
        "Proyección": "Loud",
        "Cuerpo": "Loud",
        "Claridad": "Sharpness",
        "Claridad": "Roughness",
        "Equilibrio": "L/M (Global)",
    }

    # ¿Qué "Audio X" de la encuesta corresponde a qué "Grupo" o "Clase" del CSV?
    diccionario_audios = {
        1: "Guitarra_Uxia",  # Audio 1 = Nombre en el CSV
        2: "Guitarra_Alejandro",  # Audio 2 = Nombre en el CSV
        # ... añadir el resto ...
    }
    # =====================================================================

    # --- PREPARAR DATOS SUBJETIVOS (Encuesta) ---
    print("Procesando datos subjetivos...")
    # Usamos tu función (asumo que la pegas en este script o la importas)
    df_subj, es_ranking = preparar_datos(ruta_excel_subj)

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

    # --- B. PREPARAR DATOS OBJETIVOS (CSV) ---
    print("Procesando datos objetivos...")
    df_obj = pd.read_csv(ruta_csv_obj)

    # Aquí asumo que tienes una columna 'Grupo' o 'Clase' para identificar el audio
    # Adaptalo según cómo quieras agrupar. Usaré 'Clase' de ejemplo.
    columnas_obj_necesarias = list(diccionario_metricas.values())
    medias_obj = df_obj.groupby("Clase")[columnas_obj_necesarias].mean().reset_index()

    # Normalizamos (Min-Max) los datos objetivos de 0 a 1
    for col in columnas_obj_necesarias:
        max_val = df_obj[col].max()
        min_val = df_obj[col].min()
        rango = max_val - min_val
        if rango != 0:
            medias_obj[col] = (medias_obj[col] - min_val) / rango
        else:
            medias_obj[col] = 0.0

    # --- C. GENERAR RADAR CHART POR CADA AUDIO ---
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

        # 1. Extraer y dibujar datos SUBJETIVOS (Encuesta)
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

        # 2. Extraer y dibujar datos OBJETIVOS (Código)
        datos_obj_audio = medias_obj[medias_obj["Clase"] == nombre_csv]
        if not datos_obj_audio.empty:
            valores_obj = datos_obj_audio[categorias_obj].values.flatten().tolist()
            valores_obj += valores_obj[:1]

            ax.plot(
                angulos,
                valores_obj,
                linewidth=2.5,
                linestyle="solid",
                label="Análisis Acústico (Script)",
                color="#2980b9",
            )
            ax.fill(angulos, valores_obj, color="#2980b9", alpha=0.25)

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


# Aquí deberías pegar tu función preparar_datos(ruta_excel) para que el script funcione,
# o importarla desde tu otro archivo.

if __name__ == "__main__":
    generar_radar_comparativo()
