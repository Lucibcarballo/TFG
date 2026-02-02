import os
import yaml
import pandas as pd
import numpy as np
import soundfile as sf

from caracteristicas_audio import generate_table


def load_and_calibrate_pandereta(folder_path):
    # buscamos archivos
    files = os.listdir(folder_path)
    wav_files = [f for f in files if f.endswith(".wav")]
    yaml_file = "info.yaml" if "info.yaml" in files else None

    if not wav_files or not yaml_file:
        return None, None, None

    # tomamos el primer wav por defecto
    filename = wav_files[0]
    audio_path = os.path.join(folder_path, filename)
    y, fs = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Normalizar primero a 1.0 (Peak) para tener una base limpia
    # Así nos aseguramos de que la calibración sea exacta
    y = y / (np.max(np.abs(y)) + 1e-10)

    # Leer calibración del YAML
    nivel_target = 0
    if yaml_file:
        try:
            with open(os.path.join(folder_path, yaml_file), "r") as f:
                info = yaml.safe_load(f)
            # Buscamos 'nivel' en el yaml
            nivel_target = info.get("nivel", 0)
        except Exception as e:
            print(f"Error leyendo yaml: {e}")

    # APLICAR CALIBRACIÓN CORRECTA (dB SPL -> Pascales)
    # Referencia: 20 micropascales (2e-5)
    if nivel_target > 0:
        presion_pascals = 2e-5 * (10 ** (nivel_target / 20))
        y = y * presion_pascals
        print(f"[INFO] Audio '{filename}' calibrado a {nivel_target} dB")
    else:
        print(f"[ERROR] Sin calibración en YAML. Usando nivel original normalizado.")

    return y, fs, filename


def main():
    root_dir = r"C:\Users\lucib\Desktop\TFG\audio\panderetas"
    dataset = []

    # recorremos carpetas (cada marca)
    for marca in os.listdir(root_dir):
        marca_path = os.path.join(root_dir, marca)

        if os.path.isdir(marca_path):
            print(f"Procesando marca: {marca}...")
            y, fs, filename = load_and_calibrate_pandereta(marca_path)

            if y is None:
                continue

            # extraemos caraacterísticas
            try:
                features = caracteristicas_audio.get_summary_values(y, fs)

                # Añadimos metadatos
                features["filename"] = filename
                features["clase"] = marca  # La carpeta es la "marca" o clase

                dataset.append(features)
                print(f"[OK] {filename}")
            except Exception as e:
                print(f"[ERROR] Fallo al procesar {filename}: {e}")

    # Guardar y generar tablas
    df = pd.DataFrame(dataset)

    if df.empty:
        print("No se generaron datos. Revisa las rutas.")
        return

    # Limpiar Nombres
    df["filename"] = df["filename"].str.replace(".wav", "", regex=False)
    df["filename"] = df["filename"].str.replace("_", "-")  # Evitar errores LaTeX

    nombres_cortos = {
        "filename": "Archivo",
        "clase": "Marca",
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

    # Reordenar columnas
    cols_fijas = ["Archivo", "Marca"]
    cols_resto = [c for c in df.columns if c not in cols_fijas]
    df = df[cols_fijas + cols_resto]

    df.to_csv("datasetPanderetas.csv", index=False)
    generate_table(df, "tablasPanderetas.tex")
    print("[OK] Tablas de panderetas generadas.")


if __name__ == "__main__":
    main()
