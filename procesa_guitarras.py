import os
import pandas as pd

import caracteristicas_audio


def main():
    # configuración de rutas
    input_folder = r"C:\Users\lucib\Desktop\TFG\audio\MIS_AUDIOS\notas_separadas"
    output_csv = "dataset_guitarras.csv"
    output_latex = "tabla_guitarras.tex"

    if not os.path.exists(input_folder):
        print(f"Error: No se encuentra la carpeta {input_folder}")
        return

    archivos = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    dataset = []

    print(f"Procesando {len(archivos)} notas de guitarra...")

    for archivo in archivos:
        path_completo = os.path.join(input_folder, archivo)

        # normaliza automaticamente el audio
        y, fs = caracteristicas_audio.load_audio(path_completo)

        # extracción de características resumidas
        features = caracteristicas_audio.get_summary_values(y, fs)

        # añadir metadatos
        features["filename"] = archivo
        # Etiquetamos según el nombre del archivo
        features["clase"] = (
            "electrica" if "electrica" in archivo.lower() else "española"
        )

        dataset.append(features)
        print(f"[OK] Procesado: {archivo}")

    df = pd.DataFrame(dataset)

    # Limpiar nombres de archivo para LaTeX (quitar guiones bajos)
    df["filename"] = df["filename"].str.replace(".wav", "", regex=False)
    df["filename"] = df["filename"].str.replace("notas_", "", regex=False)
    df["filename"] = df["filename"].str.replace("_", "-")

    nombres_cortos = {
        "filename": "Archivo",
        "clase": "Clase",
        "attack_time": "Atk(s)",
        "decay_time": "Dec(s)",
        "sustain_time": "Sus(s)",
        "inharmonicity": "Inharm",
        "brillantez": "Brillo",
        "low_mid_ratio": "L/M Ratio",
        "loudness": "Loud",
        "sharpness": "Sharp",
        "roughness": "Rough",
        "tnr": "TNR",
        "pr": "PR",
    }

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

    print("\n--- Generando Gráficos ---")
    caracteristicas_audio.generate_comparative_graphs(df)


if __name__ == "__main__":
    main()
