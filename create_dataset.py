import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
import librosa
from scipy.signal import get_window, resample_poly, spectrogram, find_peaks
import mosqito as mq
import seaborn as sns


# funciones sacadas de caracteristicas_audio.py adaptadas para este script


def load_audio(path):
    y, fs = sf.read(path)

    if y.ndim > 1:  # Convertir a mono si es estéreo
        y = y.mean(axis=1)

    if y.dtype != np.float32:  # normalizar si no es float32
        y = y / np.max(np.abs(y))

    return y, fs


def ADSR_curve(y, fs, n_fft=2048, hop_length=256):
    # nftt: tamaño de ventana para STFT (short-time fourier transform)
    # hop_length: salto entre ventanas

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    env = np.max(S, axis=0)

    # Evitar división por cero si es silencio
    max_val = np.max(env)
    if max_val > 0:
        env = env / max_val  # normalizar entre 0 y 1

    # Encontrar índices aproximados de ADSR
    attack_idx = np.argmax(env >= 0.9)
    decay_idx = attack_idx + np.argmax(env[attack_idx:] <= 0.5)
    sustain_idx = decay_idx + np.argmax(env[decay_idx:] <= 0.5)
    release_idx = len(env) - 1

    times = np.arange(len(env)) * hop_length / fs

    return {
        "env": env,
        "times": times,
        "attack_time": times[attack_idx],
        "decay_time": times[decay_idx],
        "sustain_time": times[sustain_idx],
        "release_time": times[release_idx],
    }


def plot_adsr(adsr):
    plt.figure(figsize=(10, 4))
    plt.plot(adsr["times"], adsr["env"], label="Envelope")
    plt.axvspan(0, adsr["attack_time"], color="green", alpha=0.3, label="Attack")
    plt.axvspan(
        adsr["attack_time"],
        adsr["decay_time"],
        color="orange",
        alpha=0.3,
        label="Decay",
    )
    plt.axvspan(
        adsr["decay_time"],
        adsr["sustain_time"],
        color="blue",
        alpha=0.3,
        label="Sustain",
    )
    plt.axvspan(
        adsr["sustain_time"],
        adsr["release_time"],
        color="red",
        alpha=0.3,
        label="Release",
    )
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud normalizada")
    plt.title("Curva ADSR")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_FFT(y, fs):
    n_samples = len(y)
    fft_vals = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n_samples, 1 / fs)
    mag = np.abs(fft_vals)
    return freqs, mag


def compute_spectrogram(y, fs, nperseg=1024, noverlap=512):
    f, t, Sxx = spectrogram(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    # Calculamos picos para la inarmonicidad
    avg_spectrum = np.mean(Sxx, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)

    peaks, _ = find_peaks(
        avg_db,
        prominence=2,
        height=np.max(avg_db) - 40,
        distance=10,
    )

    # quitamos los plt.show() para que no se abran tantas ventanas
    return f, t, Sxx, peaks


def compute_inharmonicity(peaks, f):
    if len(peaks) < 2:
        return 0  # Si no hay suficientes picos

    fp = f[peaks]  # frecuencias de los picos
    Num = len(fp)  # número de armónicos que consideramos
    Fundamental = fp[0]  # asumimos el primer pico como fundamental
    armonicos = np.arange(1, Num + 1)

    B_vals = ((fp[:Num] / (armonicos * Fundamental)) ** 2 - 1) / (armonicos**2)
    B = np.mean(np.abs(B_vals))

    return B


def subband_energy(y, fs, N, freqs, mag):
    f_min, f_max = freqs.min(), freqs.max()
    cutoffs = np.linspace(f_min, f_max, N + 1)

    band_energies = []
    for i in range(N):
        band_mask = (freqs >= cutoffs[i]) & (freqs < cutoffs[i + 1])
        band_energy = np.sum(mag[band_mask])
        band_energies.append(band_energy)

    return band_energies


def extract_mosqito_features_values(y, fs):
    resultados = {}

    # Resample necesario para mosqito
    if fs != 48000:
        y_48k = resample_poly(y, 48000, fs)
        fs_mq = 48000
    else:
        y_48k = y
        fs_mq = fs

    #  para silencios
    if np.max(np.abs(y_48k)) < 1e-5:
        return {"loudness": 0, "sharpness": 0, "roughness": 0, "tnr": 0}

    try:
        # LOUDNESS
        N, N_spec, bark_axis = mq.loudness_zwst(y_48k, fs_mq)
        resultados["loudness"] = N

        # SHARPNESS
        S = mq.sharpness_din_from_loudness(N, N_spec)
        resultados["sharpness"] = S

        # ROUGHNESS
        R, R_specific, bark, time = mq.roughness_dw(y_48k, fs_mq, overlap=0)
        resultados["roughness"] = np.mean(R) if isinstance(R, (list, np.ndarray)) else R

        # TONALITY (TNR)
        t_tnr, tnr, prom, tones_freqs = mq.tnr_ecma_st(y_48k, fs_mq)
        resultados["tnr"] = (
            np.nanmean(t_tnr)
            if np.ndim(t_tnr) > 0
            else (t_tnr if t_tnr is not None else 0)
        )

    except Exception as e:
        print(f"Error calculando mosqito: {e}")
        resultados = {"loudness": 0, "sharpness": 0, "roughness": 0, "tnr": 0}

    return resultados


def generate_table(df, output_latex):  # genera tabla para copiar y pegar en latex
    columns_display = [
        "filename",
        "clase",
        "loudness",
        "sharpness",
        "roughness",
        "brillantez",
        "inharmonicity",
    ]
    table = df[columns_display].copy()

    table["filename"] = (
        table["filename"].astype(str).str.replace("_", "\\_", regex=False)
    )  # para que latex no interprete _ como subíndice

    columns_names = {
        "filename": "Archivo",
        "clase": "Clase",
        "loudness": "Loudness",
        "sharpness": "Sharpness",
        "roughness": "Roughness",
        "brillantez": "Brillantez",
        "inharmonicity": "Inarmonicidad",
    }
    table.rename(columns=columns_names, inplace=True)

    num_columns = len(table.columns)
    column_format = "l" + "c" * (
        num_columns - 1
    )  # primera columna izq, resto centradas

    latex_code = table.to_latex(
        index=False,
        float_format="%.3f",
        caption="Características extraídas.",
        label="tab:mis_datos",
        column_format=column_format,
        position="htbp",
    )

    # guardamos en el archivo
    with open(output_latex, "w", encoding="utf-8") as f:
        f.write(latex_code)  # esto es lo que tenemos que copiar y pegar en el latex

    print(f"Tabla para LaTeX generada en: {output_latex}")


def visualizar_todo(
    output_csv,
):  # nos permite visualizar gráficos del dataset generado como .csv

    if not os.path.exists(output_csv):
        print(f"No se encuentra el archivo: {output_csv}")
        return

    df = pd.read_csv(output_csv)

    # columnas que nos interesan (excluyendo nombre de archivo y clase)
    cols_interes = [
        "loudness",
        "sharpness",
        "roughness",
        "tnr",
        "brillantez",
        "inharmonicity",
        "attack_time",
        "decay_time",
        "sustain_time",
    ]

    cols_reales = [c for c in cols_interes if c in df.columns]

    sns.set_theme(style="whitegrid")  # configuración estética

    # _____________________Boxlplots_________________________

    n_cols = 3
    n_rows = (len(cols_reales) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_reales):
        # El gráfico mágico: x=Clase, y=Característica
        sns.boxplot(data=df, x="clase", y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"Distribución de {col}", fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Valor")

    # Borrar ejes vacios si sobran
    for i in range(len(cols_reales), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig("comparacion_boxplots.png", dpi=300)
    print("[INFO] Guardado: comparacion_boxplots.png")
    plt.show()

    # ___________________PAIRPLOT (Relaciones entre variables)_____________________

    cols_pairplot = [
        "clase",
        "loudness",
        "sharpness",
        "roughness",
        "tnr",
        "brillantez",
        "inharmonicity",
        "attack_time",
        "decay_time",
        "sustain_time",
    ]
    cols_pairplot = [c for c in cols_pairplot if c in df.columns]

    pair_plot = sns.pairplot(
        df[cols_pairplot], hue="clase", palette="Set1", diag_kind="kde"
    )
    pair_plot.fig.suptitle("Relación entre variables (Scatter Matrix)", y=1.02)

    pair_plot.savefig("relaciones_pairplot.png", dpi=300)
    print("[INFO] Guardado: relaciones_pairplot.png")
    plt.show()

    # ______________________MATRIZ DE CORRELACIÓN_________________________
    plt.figure(figsize=(10, 8))

    # Calculamos correlación solo de numéricas
    corr = df[cols_reales].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación entre Características")

    plt.tight_layout()
    plt.savefig("matriz_correlacion.png", dpi=300)
    print("[INFO] Guardado: matriz_correlacion.png")
    plt.show()


def main():
    input_folder = "C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/notas_separadas/"
    output_csv = "C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/dataset_final.csv"
    output_latex = "C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/tabla_resultados.tex"

    if not os.path.exists(input_folder):
        print("No existe la carpeta de notas separadas.")
        return

    archivos = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    dataset = []

    print(f"Procesando {len(archivos)} notas...")

    for i, archivo in enumerate(archivos):
        path = os.path.join(input_folder, archivo)
        y, fs = load_audio(path)

        # ____________________Extraer características

        # ADSR
        adsr = ADSR_curve(y, fs)
        # plot_adsr(adsr)

        # Espectrales e Inarmonicidad
        freqs, mag = compute_FFT(y, fs)
        f, t, Sxx, peaks = compute_spectrogram(y, fs)
        inharmonicity = compute_inharmonicity(peaks, f)

        # Bandas (Brillantez)
        # Usamos N=3 bandas para poder calcular brillantez como tenías
        band_energies = subband_energy(y, fs, 3, freqs, mag)
        # Evitar div por cero
        e_low = band_energies[0]
        e_mid = band_energies[1] if band_energies[1] > 0 else 1e-10
        e_high = band_energies[2]

        brillantez = e_high / e_mid
        il_bajo_medio = e_low / e_mid

        # Mosqito
        mosqito_feats = extract_mosqito_features_values(y, fs)

        # Construir la fila del dataset
        fila = {
            "filename": archivo,
            # Etiqueta
            "clase": "electrica" if "electrica" in archivo else "española",
            # Datos ADSR
            "attack_time": adsr["attack_time"],
            "decay_time": adsr["decay_time"],
            "sustain_time": adsr["sustain_time"],
            # Datos Espectrales
            "inharmonicity": inharmonicity,
            "brillantez": brillantez,
            "indice_low_mid": il_bajo_medio,
            "energy_total": np.sum(mag),
            # Datos Mosqito
            "loudness": mosqito_feats["loudness"],
            "sharpness": mosqito_feats["sharpness"],
            "roughness": mosqito_feats["roughness"],
            "tnr": mosqito_feats["tnr"],
        }

        dataset.append(fila)
        print(f"[{i+1}/{len(archivos)}] Procesado: {archivo}")

    # Guardar CSV y generar LaTeX
    df = pd.DataFrame(dataset)
    df.to_csv(output_csv, index=False)

    print("\nGenerando tabla para LaTeX...")
    generate_table(df, output_latex)

    print("\n" + "=" * 30)
    print("¡Listo!")
    print(f"1. Datos completos en: {output_csv}")
    print(f"2. Tabla visual para TFG en: {output_latex}")
    print("=" * 30)

    visualizar_todo(output_csv)

    # # _______________________________Guardar dataset csv_______________________________
    # df = pd.DataFrame(dataset)
    # df.to_csv(output_csv, index=False)

    # print("\n" + "=" * 30)
    # print("Dataset Generado con Éxito")
    # print(df.head())
    # print(f"Guardado en: {output_csv}")

    # print(f"Tamaño total del dataset: {len(df)}")


if __name__ == "__main__":
    main()
