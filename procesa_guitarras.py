import os
import pandas as pd
import math

import caracteristicas_audio


def main():
    # _____________________OJO: AJUSTAR configuración de rutas_________________________
    input_folder = r"C:\Users\lucib\Desktop\TFG\codigo\docs\notas"
    output_csv = "dataset_guitarras_grabaciones_notas.csv"
    output_latex = "tabla_guitarras_grabaciones_notas.tex"
    # __________________________________________________________________________________

    if not os.path.exists(input_folder):
        print(f"Error: No se encuentra la carpeta {input_folder}")
        return

    archivos = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    dataset = []

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
        clase_asignada = "Uxía" if "uxia" in archivo.lower() else "Alejandro"

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

                note_features = caracteristicas_audio.get_global_features(y_note, fs)

                # combinamos características globales y de nota
                features = {**global_features, **note_features}

                # etiquetar con sufijo de nota
                features["filename"] = f"{nombre_base}_nota{i+1}"
                features["clase"] = clase_asignada

                dataset.append(features)

        else:
            # si no es un archivo de notas, procesamos el audio completo
            print(
                f"[{archivo}] No es un archivo de notas. Evaluando solo métricas globales..."
            )

            features = global_features.copy()

            features["filename"] = nombre_base
            features["clase"] = clase_asignada

            dataset.append(features)

        print(f"[OK] Procesado: {archivo}")

    df = pd.DataFrame(dataset)

    print("\n--- [DEBUG] 1. COLUMNAS ORIGINALES AL EXTRAER EL AUDIO ---")
    print(df.columns.tolist())

    # Limpiar nombres de archivo para LaTeX (quitar guiones bajos)
    df["filename"] = df["filename"].str.replace(".wav", "", regex=False)
    df["filename"] = df["filename"].str.replace("notas_", "", regex=False)
    df["filename"] = df["filename"].str.replace("_", "-")

    # renombramos para el csv y la tabla LaTeX, graficos usan los nombres originales
    nombres_cortos = {
        "filename": "Archivo",
        "clase": "Clase",
        "attack_time": "Atk(s)",
        "decay_time": "Dec(s)",
        "sustain_time": "Sus(s)",
        "inharmonicity": "Inharm",
        "brillantez_nota": "Brillo (Nota)",
        "low_mid_ratio_nota": "L/M (Nota)",
        "brillantez_global": "Brillo (Global)",
        "low_mid_ratio_global": "L/M (Global)",
        "loudness": "Loud",
        "sharpness": "Sharp",
        "roughness": "Rough",
        "tnr": "TNR",
        "pr": "PR",
    }
    print(df.columns.tolist())

    df.rename(columns=nombres_cortos, inplace=True)

    # Primero definimos las fijas
    cols_fijas = ["Archivo", "Clase"]
    # Luego cogemos el resto de columnas que haya en el DataFrame (excepto las fijas)
    cols_resto = [c for c in df.columns if c not in cols_fijas]
    # Concatenamos para el orden final
    df = df[cols_fijas + cols_resto]

    df.to_csv(output_csv, index=False)

    # Generar Tablas LaTeX
    caracteristicas_audio.generate_table(df, output_latex)

    print(f"\n Resultados en {output_csv} y {output_latex}")

    print("\n--- Generando gráficos ---")
    caracteristicas_audio.generate_comparative_graphs(df)
    caracteristicas_audio.generate_small_multiples_bars(
        df, filename="small_multiples.png"
    )

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()
