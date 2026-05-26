import os
import pandas as pd
import math

import caracteristicas_audio


def main():
    # _____________________OJO: AJUSTAR configuración de rutas_________________________
    # input_folder = r"C:\Users\lucib\Desktop\TFG\codigo\docs\piezas"

    # output_csv_notas = "dataset_guitarras_grabaciones_notas.csv"
    # output_csv_global = "dataset_guitarras_grabaciones_global.csv"

    # output_latex_notas = "tabla_guitarras_grabaciones_notas.tex"
    # output_latex_global = "tabla_guitarras_grabaciones_global.tex"

    input_folder = r"C:\Users\lucib\Desktop\TFG\codigo\audios\notas"

    output_csv_notas = "dataset_guitarras_grabaciones_notas.csv"
    output_csv_global = "dataset_guitarras_grabaciones_global.csv"

    output_latex_notas = "tabla_guitarras_grabaciones_notas.tex"
    output_latex_global = "tabla_guitarras_grabaciones_global.tex"

    # __________________________________________________________________________________

    if not os.path.exists(input_folder):
        print(f"Error: No se encuentra la carpeta {input_folder}")
        return

    archivos = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

    dataset_notas = []
    dataset_global = []

    print(f"Procesando {len(archivos)} audios de guitarra...")

    for archivo in archivos:
        path_completo = os.path.join(input_folder, archivo)

        # normaliza automaticamente el audio
        y_full, fs = caracteristicas_audio.load_audio(path_completo)

        print(f"Extrayendo características globales de {archivo}...")

        # extracción de características resumidas
        global_features = caracteristicas_audio.get_global_features(y_full, fs)
        # preparar metadatos base
        nombre_base = archivo.replace(".wav", "")
        clase_asignada = "Intérprete 2" if "uxia" in archivo.lower() else "Intérprete 1"
        archivo_minusculas = archivo.lower()

        if "uxia" in archivo_minusculas:
            clase_asignada = "Intérprete 2"
        elif "alejandro" in archivo_minusculas:
            clase_asignada = "Intérprete 1"
        elif "ambos" in archivo_minusculas:
            clase_asignada = "Ambos intérpretes"
        else:
            clase_asignada = None

        # SIEMPRE guardamos los datos globales en el dataset global
        global_entry = global_features.copy()
        global_entry["filename"] = nombre_base
        global_entry["clase"] = clase_asignada
        dataset_global.append(global_entry)

        # DIVISION EN NOTAS si contiene "notas" en el nombre
        if "notas" in archivo.lower():
            print(f"Dividiendo {archivo} en notas individuales...")
            chunk_length = 4 * fs  # 4 segundos por nota
            # Usamos math.ceil para redondear hacia arriba y no perder el último trozo del audio de 39s
            num_chunks = math.ceil(len(y_full) / chunk_length)
            print(
                f" - {archivo} se dividirá en {num_chunks} notas de 4 segundos cada una."
            )

            for i in range(num_chunks):
                start = i * chunk_length
                end = start + chunk_length
                y_note = y_full[start:end]

                note_features = caracteristicas_audio.get_note_features(y_note, fs)

                # Renombrar las claves de la nota para que no pisen a las globales
                note_features_renombradas = {
                    f"{k}_nota": v for k, v in note_features.items()
                }

                # combinamos características globales y de nota
                features = {**global_features, **note_features_renombradas}

                # etiquetar con sufijo de nota
                features["filename"] = f"{nombre_base}_nota{i+1}"
                features["clase"] = clase_asignada

                dataset_notas.append(features)

        else:
            # si no es un archivo de notas, procesamos el audio completo
            print(
                f"[{archivo}] No es un archivo de notas. Evaluando solo métricas globales..."
            )

            features = global_features.copy()

            features["filename"] = nombre_base
            features["clase"] = clase_asignada

            dataset_global.append(features)

        print(f"[OK] Procesado: {archivo}")

    df_notas = pd.DataFrame(dataset_notas)
    df_global = pd.DataFrame(dataset_global)

    def procesar_dataframe(df_temp):
        if df_temp.empty:
            return df_temp  # Devuelve el DataFrame vacío sin procesar

        # Limpiar nombres de archivo para LaTeX (quitar guiones bajos)
        df_temp["filename"] = df_temp["filename"].str.replace(".wav", "", regex=False)
        df_temp["filename"] = df_temp["filename"].str.replace("notas_", "", regex=False)
        df_temp["filename"] = df_temp["filename"].str.replace("_", "-")

        # Forzamos pasar a minúsculas temporalmente solo para la comprobación y evitar fallos de mayúsculas
        df_temp["filename"] = df_temp["filename"].str.replace(
            "alejandro", "Intérprete 1", case=False, regex=False
        )
        df_temp["filename"] = df_temp["filename"].str.replace(
            "uxia", "Intérprete 2", case=False, regex=False
        )
        df_temp["filename"] = df_temp["filename"].str.replace(
            "uxía", "Intérprete 2", case=False, regex=False
        )

        # renombramos para el csv y la tabla LaTeX, graficos usan los nombres originales
        nombres_cortos = {
            "filename": "Archivo",
            "clase": "Clase",
            "attack_time_nota": "Atk(s)",
            "decay_time_nota": "Dec(s)",
            "sustain_time_nota": "Sus(s)",
            "inharmonicity_nota": "Inharm",
            "brillantez_nota": "Brillo (Nota)",
            "low_mid_ratio_nota": "L/M (Nota)",
            "brillantez_global": "Brillo (Global)",
            "low_mid_ratio_global": "L/M (Global)",
            "loudness": "Loud",
            "sharpness": "Sharp",
            "roughness": "Rough",
            # "tnr": "TNR",
            # "pr": "PR",
        }

        df_temp.rename(columns=nombres_cortos, inplace=True)

        # Primero definimos las fijas
        cols_fijas = ["Archivo", "Clase"]
        # Luego cogemos el resto de columnas que haya en el DataFrame (excepto las fijas)
        cols_resto = [c for c in df_temp.columns if c not in cols_fijas]
        # Concatenamos para el orden final
        return df_temp[cols_fijas + cols_resto]

    df_global = procesar_dataframe(df_global)
    df_notas = procesar_dataframe(df_notas)

    #  Datos Globales
    if not df_global.empty:
        df_global.to_csv(output_csv_global, index=False)
        caracteristicas_audio.generate_table(df_global, output_latex_global)
        print(f"Globales: {output_csv_global} y {output_latex_global}")

    #  Datos de Notas
    if not df_notas.empty:
        df_notas.to_csv(output_csv_notas, index=False)
        caracteristicas_audio.generate_table(df_notas, output_latex_notas)
        print(f"Notas: {output_csv_notas} y {output_latex_notas}")

        # ANALISIS POR GUITARRA - PROMEDIOS DE NOTAS
        print("\n--- Calculando promedios nota a nota por guitarra ---")
        df_promedios = df_notas.copy()

        # Extraer el nombre de la guitarra (ej: "g1-alejandro-nota1" -> "g1")
        df_promedios["Guitarra"] = df_promedios["Archivo"].apply(
            lambda x: str(x).split("-")[0]
        )

        # Definir qué columnas numéricas de las notas queremos promediar
        cols_notas_individuales = [
            "Atk(s)",
            "Dec(s)",
            "Sus(s)",
            "Inharm",
            "Brillo (Nota)",
            "L/M (Nota)",
        ]
        cols_existentes = [
            c for c in cols_notas_individuales if c in df_promedios.columns
        ]

        # Agrupar por guitarra y calcular la media
        df_guitarra_notas_medias = (
            df_promedios.groupby("Guitarra")[cols_existentes].mean().reset_index()
        )

        # nuevos csv y .tex
        output_csv_promedio_notas = "dataset_guitarras_promedio_notas.csv"
        output_latex_promedio_notas = "tabla_guitarras_promedio_notas.tex"

        df_guitarra_notas_medias.to_csv(output_csv_promedio_notas, index=False)
        caracteristicas_audio.generate_table(
            df_guitarra_notas_medias, output_latex_promedio_notas
        )

        print(f"[OK] CSV de promedios guardado: {output_csv_promedio_notas}")
        print(f"[OK] Tabla LaTeX de promedios guardada: {output_latex_promedio_notas}")

    print("\n--- Generando gráficos ---")
    caracteristicas_audio.generate_comparative_graphs(df_global)
    caracteristicas_audio.generate_small_multiples_bars(
        df_global, filename="small_multiples.png"
    )

    if not df_notas.empty:
        caracteristicas_audio.graph_notes(df_notas, filename="evolution_notes.png")

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()
