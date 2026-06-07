import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import get_window, resample_poly, spectrogram, find_peaks
import mosqito as mq
import numpy as np
import librosa
import seaborn as sns
from math import pi
import re  # Necesario para detectar patrones de texto
from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Puedes poner esto justo después de tus imports para que afecte a todos los gráficos
import matplotlib.pyplot as plt


def load_audio(path, ref_path=None):  # calibra si se proporciona nivel en dB

    y, fs = sf.read(path)

    # Si el audio es estereo, convertir a mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    if ref_path is not None:  # calibrar si damos audio de referencia de calibracion

        y_ref, _ = sf.read(ref_path)
        if y_ref.ndim > 1:
            y_ref = y_ref.mean(axis=1)

        sonometro = 69.41  # dbs medidos con sonometro

        # Calculamos energías (RMS)
        rms_ref = np.sqrt(np.mean(y_ref**2))
        rms_actual = np.sqrt(np.mean(y**2))  # del audio a calibrar

        C = sonometro - 20 * np.log10(
            max(rms_ref, 1e-10)
        )  # diferencia entre el nivel medido y el de referencia

        nivel_calibrado = (
            20 * np.log10(max(rms_actual, 1e-10)) + C
        )  # nivel del audio a calibrar

        delta_db = sonometro - nivel_calibrado
        gain = 10 ** (delta_db / 20)
        y_calibrada = y * gain  # aplicamos ganancia para ajustar al nivel de referencia

        print(f"-> El audio '{path}' original medía: {nivel_calibrado:.2f} dB SPL")

        if np.max(np.abs(y_calibrada)) > 1.0:
            print("[WARNING] El audio calibrado supera los 0 dB FS (clipping).")

        return y_calibrada, fs

    return y, fs


def ADSR_curve(y, fs, n_fft=2048, hop_length=256):
    # nfft = tamaño ventana, hop_length= cuanto se desplaza la ventana

    S = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    )  # stft: short fourier transform
    env = np.max(S, axis=0)  # máximo por muestra
    env = env / (np.max(env) + 1e-10)  # normalizar

    # calculo de tiempos en segundos
    times = np.arange(len(env)) * hop_length / fs

    start_mask = env > 0.1
    if not np.any(start_mask):  # si no hay sonido en todo el clip, todo a 0
        return {
            "env": env,
            "times": times,
            "attack_time": 0.0,
            "decay_time": 0.0,
            "sustain_time": 0.0,
        }

    strat_idx = np.argmax(start_mask)

    # ATAQUE
    attack_idx = strat_idx + np.argmax(env[strat_idx:])  # ataque hasta pico maximo

    # DECAY
    decay = env[attack_idx:] <= 0.5
    if np.any(decay):
        decay_idx = attack_idx + np.argmax(decay)
    else:
        decay_idx = len(env) - 1  # si no baja del 50%, decay hasta el final

    # SUSTAIN
    sustain = env[decay_idx:] <= 0.1
    if np.any(sustain):
        sustain_idx = decay_idx + np.argmax(sustain)
    else:
        sustain_idx = len(env) - 1  # si no baja del 10%, sustain hasta el final

    # RESULTADOS: duraciones reales
    attack_duration = times[attack_idx] - times[strat_idx]
    decay_duration = times[decay_idx] - times[attack_idx]
    sustain_duration = times[sustain_idx] - times[decay_idx]

    return {
        "env": env,
        "times": times,
        "attack_time": attack_duration,
        "decay_time": decay_duration,
        "sustain_time": sustain_duration,
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


def compute_FFT(y, fs):
    n_samples = len(y)  # num muestras
    fft_vals = np.fft.rfft(y)  # transformada rapida de fourier
    freqs = np.fft.rfftfreq(n_samples, 1 / fs)
    mag = np.abs(fft_vals)

    return freqs, mag


def plot_fft(freqs, mag):
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, mag)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Energía (log)")
    plt.title("Energía vs Frecuencia")
    plt.grid(True)
    plt.show()


def compute_spectrogram_data(y, fs, nperseg=1024, noverlap=512):
    f, t, Sxx = spectrogram(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    # pulso de energía temporal
    energy_pulse = np.sum(Sxx, axis=0)
    energy_db_time = 10 * np.log10(energy_pulse + 1e-10)

    # promedio espectral para picos
    avg_spectrum = np.mean(Sxx, axis=1)
    avg_db_freq = 10 * np.log10(avg_spectrum + 1e-10)

    peaks, _ = find_peaks(
        avg_db_freq,
        prominence=2,
        height=np.max(avg_db_freq) - 40,
        distance=10,
    )

    return {
        "f": f,
        "t": t,
        "Sxx": Sxx,
        "energy_db_time": energy_db_time,
        "avg_db_freq": avg_db_freq,
        "peaks": peaks,
    }


def plot_spectrogram_analysis(spec_data):
    # grafico pulso temporal
    plt.figure(figsize=(10, 4))
    plt.plot(spec_data["t"], spec_data["energy_db_time"], color="purple")
    plt.title("Pulso temporal de energía")
    plt.xlabel("Tiempo [s]")
    plt.grid(True)
    plt.show()

    # grafico picos espectrales
    plt.figure(figsize=(10, 4))
    plt.plot(spec_data["f"], spec_data["avg_db_freq"], color="blue")
    peaks = spec_data["peaks"]
    plt.plot(spec_data["f"][peaks], spec_data["avg_db_freq"][peaks], "ro", markersize=5)
    plt.title("Promedio espectral y armónicos")
    plt.xlabel("Frecuencia [Hz]")
    plt.grid(True)
    plt.show()

    print("Frecuencias pico detectadas:", spec_data["f"][peaks])


def compute_inharmonicity(peaks, f):
    # la inarmonicidad (B) describe cuánto se desvían los armónicos reales de un sonido respecto de los armónicos ideales

    fp = f[peaks]  # frecuencias de los picos
    Num = len(fp)  # num de armónicos que consideramos
    if Num < 2:
        return 0.0  # Si no hay suficientes armónicos, devolvemos 0

    Fundamental = fp[0]  # asumimos el primer pico como fundamental

    f_armonicos = fp[1:]  # armónicos reales (excluyendo el fundamental)
    n = np.arange(2, Num + 1)  # números de armónicos ideales (2, 3, 4, ...)

    # Coeficiente B para cada armónico
    B_vals = ((f_armonicos / (n * Fundamental)) ** 2 - 1) / (n**2)

    # devolvemos la media (un solo número)
    return np.mean(np.abs(B_vals))


def compute_subband_data(y, fs, freqs, mag):
    # energia espectral
    f_min, f_max = freqs.min(), freqs.max()
    # cutoffs = np.linspace(f_min, f_max, 4)  # 3 bandas = 4
    # cutoffs = np.array(
    #     [80, 300, 1000, 12000]
    # )  # límites segun guitarra moderna (afinación 440 Hz)
    cutoffs = np.array(
        [100, 250, 800, 10000]
    )  # limites segun guitarra barroca (afinacion 415 Hz)

    # integrar energía en cada banda
    band_energies = []
    for i in range(3):
        band_mask = (freqs >= cutoffs[i]) & (freqs < cutoffs[i + 1])
        band_energies.append(np.sum(mag[band_mask]))

    # energía temporal por banda (para Plot)
    frame_len = int(0.05 * fs)
    hop = int(0.025 * fs)
    window = get_window("hann", frame_len)
    # bands_def = {
    #     "graves": (20, 200),
    #     "medios": (200, 2000),
    #     "agudos": (2000, fs / 2 - 100),
    # }
    # bands_def = {
    #     "graves": (80, 300),
    #     "medios": (300, 1000),
    #     "agudos": (1000, 12000),
    # }
    bands_def = {
        "graves": (100, 250),
        "medios": (250, 800),
        "agudos": (800, 10000),
    }

    n_frames = int((len(y) - frame_len) / hop) + 1
    temporal_energy = {name: np.zeros(n_frames) for name in bands_def}

    if n_frames > 0:
        freqs_frame = np.fft.rfftfreq(frame_len, 1 / fs)
        for i in range(n_frames):
            frame = y[i * hop : i * hop + frame_len] * window
            mag2 = np.abs(np.fft.rfft(frame)) ** 2

            for name, (f_low, f_high) in bands_def.items():
                mask = (freqs_frame >= f_low) & (freqs_frame < f_high)
                temporal_energy[name][i] = np.sum(mag2[mask])

    t_frames = np.arange(n_frames) * hop / fs

    return {
        "band_energies": band_energies,  # [Low, Mid, High]
        "cutoffs": cutoffs,
        "temporal_energy": temporal_energy,
        "t_frames": t_frames,
    }


def plot_subband_analysis(sub_data, freqs, mag):
    # division espectral
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mag)
    for c in sub_data["cutoffs"]:
        plt.axvline(c, color="r", linestyle="--")
    plt.title("División en bandas de frecuencia")
    plt.show()

    # Evolucion temporal
    plt.figure(figsize=(10, 6))
    for i, (name, energy) in enumerate(sub_data["temporal_energy"].items(), 1):
        plt.subplot(3, 1, i)
        plt.plot(sub_data["t_frames"], energy / (np.max(energy) + 1e-10), color=f"C{i}")
        plt.title(f"Subbanda {name}")
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_mosqito_data(y, fs):
    # Resample a 48kHz para Mosqito
    y_48 = resample_poly(y, 48000, fs) if fs != 48000 else y
    fs_mq = 48000

    results = {}

    # Loudness
    N, N_spec, bark_axis = mq.loudness_zwst(y_48, fs_mq)
    results["loudness"] = {"val": N, "spec": N_spec, "bark": bark_axis}

    # Sharpness
    S = mq.sharpness_din_from_loudness(N, N_spec)
    # Sharpness temporal
    S_tv, t_tv = mq.sharpness_din_tv(y_48, fs_mq, skip=0.1)
    results["sharpness"] = {"val": S, "tv": S_tv, "time": t_tv}

    # Roughness
    R, R_spec, bark_r, time_r = mq.roughness_dw(y_48, fs_mq, overlap=0)
    results["roughness"] = {"val": np.mean(R), "spec": R_spec, "bark": bark_r}

    # Tonality (TNR y PR)
    t_tnr, tnr, _, tones_freqs_tnr = mq.tnr_ecma_st(y_48, fs_mq)
    t_pr, pr, _, tones_freqs_pr = mq.pr_ecma_st(y_48, fs_mq)

    results["tonality"] = {
        "tnr_global": np.nanmean(t_tnr),
        "tnr_spec": tnr,
        "tnr_freqs": tones_freqs_tnr,
        "pr_global": np.nanmean(t_pr),
        "pr_spec": pr,
        "pr_freqs": tones_freqs_pr,
    }

    return results


def plot_mosqito_analysis(mq_data):
    # Loudness
    d = mq_data["loudness"]
    plt.figure()
    plt.plot(d["bark"], d["spec"])
    plt.title(f"Loudness Spectrum (N={d['val']:.2f} sone)")
    plt.xlabel("Bark")
    plt.show()

    # Sharpness Time
    d = mq_data["sharpness"]
    plt.figure()
    plt.plot(d["time"], d["tv"])
    plt.title("Sharpness vs Time")
    plt.show()

    # Roughness
    d = mq_data["roughness"]
    plt.figure()
    plt.plot(d["bark"], d["spec"])
    plt.title(f"Roughness Spectrum (R={d['val']:.2f} asper)")
    plt.show()


def get_note_features(y, fs):
    adsr = ADSR_curve(y, fs)

    # inarmonicidad
    spec = compute_spectrogram_data(y, fs)
    inharm = compute_inharmonicity(spec["peaks"], spec["f"])

    freqs, mag = compute_FFT(y, fs)
    energy_spectrum = mag**2  # energia es proportional al cuadrado de la magnitud

    # brillantez nota a nota
    picos = spec["peaks"]

    if len(picos) > 1:
        f_picos = spec["f"][picos]  # frecuencias en Hz de los picos

        idx_fundamental = np.argmin(
            np.abs(freqs - f_picos[0])
        )  # primer pico (fundamental)

        # Sumamos la energía dando un pequeño margen de 3 bins para capturar la "campana" entera del pico
        margen = 3
        e_fund = np.sum(
            energy_spectrum[
                max(0, idx_fundamental - margen) : min(
                    len(energy_spectrum), idx_fundamental + margen + 1
                )
            ]
        )

        # energia de los armonicos
        e_harm = 0
        for f_harm in f_picos[1:]:
            idx_h = np.argmin(np.abs(freqs - f_harm))
            e_harm += np.sum(
                energy_spectrum[
                    max(0, idx_h - margen) : min(
                        len(energy_spectrum), idx_h + margen + 1
                    )
                ]
            )

        # Ratio: Energía en los armónicos / Energía en la fundamental
        brillo_nota = e_harm / (e_fund + 1e-10)

    else:
        brillo_nota = 0.0  # Si no hay armónicos, el brillo es 0

    sub = compute_subband_data(y, fs, freqs, mag)
    low_mid_nota = sub["band_energies"][0] / (sub["band_energies"][1] + 1e-10)

    return {
        "attack_time": adsr["attack_time"],
        "decay_time": adsr["decay_time"],
        "sustain_time": adsr["sustain_time"],
        "inharmonicity": inharm,
        "brillantez": brillo_nota,
        "low_mid_ratio": low_mid_nota,
    }


def get_global_features(y, fs):

    freqs, mag = compute_FFT(y, fs)
    sub = compute_subband_data(y, fs, freqs, mag)
    e = sub["band_energies"]
    brillo_global = e[2] / (e[1] + 1e-10)
    low_mid_global = e[0] / (e[1] + 1e-10)

    # Análisis psicoacústico con Mosqito
    mq_data = compute_mosqito_data(y, fs)

    return {
        "brillantez_global": brillo_global,
        "low_mid_ratio_global": low_mid_global,
        "loudness": mq_data["loudness"]["val"],
        "sharpness": mq_data["sharpness"]["val"],
        "roughness": mq_data["roughness"]["val"],
        # "tnr": mq_data["tonality"]["tnr_global"],
        # "pr": mq_data["tonality"]["pr_global"],
    }


def generate_table(df, output_filename):  # genera tabla LaTeX
    try:
        # Redondeamos decimales
        latex_code = df.round(3).to_latex(
            index=False, caption=f"Datos de {output_filename}"
        )
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(latex_code)
        print(f"Tabla LaTeX guardada en {output_filename}")
    except Exception as e:
        print(f"Error generando tabla LaTeX: {e}")


def generate_table(df, output_filename, landscape=False):  # para tabla latex

    try:
        latex_code = df.to_latex(
            index=False,
            float_format="%.3f",  # Redondear a 3 decimales
            caption=f"Datos de {output_filename}",
            escape=False,
        )

        latex_code = latex_code.replace(
            "\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}"
        )
        latex_code = latex_code.replace("\\end{tabular}", "\\end{tabular}%\n}")

        # opción de girar la hoja (Landscape)
        if landscape:
            latex_code = "\\begin{landscape}\n" + latex_code + "\n\\end{landscape}"

        # guardar archivo
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(latex_code)

        print(f"Tabla LaTeX guardada en {output_filename} (Landscape={landscape})")

    except Exception as e:
        print(f"Error generando tabla LaTeX: {e}")


def limpiar_nombre(nombre):
    # Regex: Busca guiones + "nota" + guiones + digitos al final
    # Ej: "pua-electrica-nota-1" -> "pua-electrica"
    # Ej: "guitarra_05" -> "guitarra"
    return re.sub(r"[-_]?(nota)?[-_]?\d+$", "", nombre, flags=re.IGNORECASE)


def generate_comparative_graphs(
    df,
):  # recibe el dataframe con las características de todas las notas y genera gráficos comparativos entre clases

    sns.set_style("whitegrid")

    # Creamos una columna temporal 'Grupo' con el nombre limpio
    df["Grupo"] = df["Archivo"].apply(limpiar_nombre)

    cols_a_excluir = [
        "Atk(s)",
        "Dec(s)",
        "Sus(s)",
        "Inharm",
        "Brillo (Nota)",
        "L/M (Nota)",
        "Numero_Nota",
        "Cuerda",
    ]  # excluimos características nota a nota para centrarnos en las globales por archivo

    # Identificamos columnas numéricas y filtramos
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_numericas = [col for col in cols_numericas if col not in cols_a_excluir]

    # calculamos la media de todas las notas que tengan el mismo nombre de Grupo
    # Mantenemos 'Clase' (ej: electrica vs española)
    df_agrupado = df.groupby(["Grupo", "Clase"])[cols_numericas].mean().reset_index()

    print(
        f"Audios agrupados: de {len(df)} archivos originales a {len(df_agrupado)} grupos promedio."
    )

    # NORMALIZACIÓN para representación gráfica (0-1 por característica)
    df_norm = df_agrupado.copy()
    for col in cols_numericas:
        max_val = df_norm[col].abs().max()  # abs por si hubiera valores negativos
        if max_val != 0:
            df_norm[col] = df_norm[col] / max_val
        else:
            df_norm[col] = 0.0

    # Configuración de colores
    clases_unicas = df_norm["Clase"].unique()
    paleta = sns.color_palette("Set2", len(clases_unicas))
    mapa_colores = dict(zip(clases_unicas, paleta))

    # _______________________RADAR CHART_______________________
    df_radar = df_norm.groupby("Clase")[cols_numericas].mean().reset_index()

    categorias = cols_numericas
    N = len(categorias)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categorias, size=16)
    ax.tick_params(axis="y", colors="gray", labelsize=16)  # nums del radar en gris

    for i, row in df_radar.iterrows():
        valores = row[cols_numericas].values.flatten().tolist()
        valores += valores[:1]
        nombre_clase = row["Clase"]
        color = mapa_colores.get(nombre_clase, "black")
        ax.plot(
            angles,
            valores,
            linewidth=2.5,
            linestyle="solid",
            label=nombre_clase,
            color=color,
        )
        ax.fill(angles, valores, color=color, alpha=0.1)

    plt.title(
        "Radar Chart de características promedio por intérprete",
        y=1.08,
        fontsize=18,
        fontweight="bold",
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=16,
    )

    plt.savefig("radar_chart.png", dpi=300, bbox_inches="tight")
    print("Guardado: radar_chart.png")
    plt.close()

    # # _____________________BARRAS AGRUPADAS_______________________
    # print("Generando gráfico de barras...")

    # # Usamos df_norm que ya tiene los grupos (pua-electrica, uña-electrica, etc.)
    # df_melt = df_norm.melt(
    #     id_vars=["Grupo", "Clase"],
    #     value_vars=cols_numericas,
    #     var_name="Característica",
    #     value_name="Valor Normalizado",
    # )

    # # Ordenar para que salgan agrupados visualmente por clase
    # df_melt.sort_values(by=["Clase", "Grupo"], inplace=True)

    # num_grupos = len(df_norm["Grupo"].unique())
    # altura_figura = max(5, num_grupos * 0.8)

    # plt.figure(figsize=(12, altura_figura))

    # sns.barplot(
    #     data=df_melt,
    #     y="Grupo",
    #     x="Valor Normalizado",
    #     hue="Característica",
    #     palette="tab10",
    #     orient="h",
    # )

    # plt.title("Características promedio por grupos")
    # plt.xlabel("Magnitud normalizada")
    # plt.ylabel("Grupo")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Características")
    # plt.tight_layout()
    # plt.grid(axis="x", alpha=0.3)

    # plt.savefig("grafico_barras.png", dpi=300, bbox_inches="tight")
    # print("Guardado: grafico_barras.png")

    # plt.close()


def generate_3d_pca_graph(
    df, filename="grafico_3d_pca.png"
):  # para visualizar la separación entre pua, uña, yema
    # utilizamos PCA para reducir dimensionalidad a 3D y graficar

    def detectar_tecnica(nombre_archivo):
        nombre = nombre_archivo.lower()
        if "pua" in nombre:
            return "Púa"
        if "uña" in nombre:
            return "Uña"
        if "yema" in nombre:
            return "Yema"
        return "Otra"

    df_pca = df.copy()
    df_pca["Tecnica"] = df_pca["Archivo"].apply(detectar_tecnica)

    # Filtrar solo las técnicas de interés
    df_pca = df_pca[df_pca["Tecnica"].isin(["Púa", "Uña", "Yema"])].copy()

    if df_pca.empty:
        print("Error: No hay datos suficientes de Púa/Uña/Yema.")
        return

    cols_features = df_pca.select_dtypes(include=[np.number]).columns.tolist()
    X = df_pca[cols_features].values

    X_scaled = StandardScaler().fit_transform(
        X
    )  # misma escala para todas las características

    pca = PCA(n_components=3)  # para reducir a 3 dimensiones
    componentes_principales = pca.fit_transform(X_scaled)

    df_plot = pd.DataFrame(data=componentes_principales, columns=["PC1", "PC2", "PC3"])
    df_plot["Tecnica"] = df_pca["Tecnica"].values  # Recuperamos las etiquetas

    varianza = pca.explained_variance_ratio_
    print(
        f"Varianza explicada: PC1={varianza[0]:.2f}, PC2={varianza[1]:.2f}, PC3={varianza[2]:.2f}"
    )
    print(f"Total información retenida: {sum(varianza)*100:.1f}%")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colores_tecnica = {"Púa": "#e74c3c", "Uña": "#3498db", "Yema": "#2ecc71"}

    for tecnica, grupo in df_plot.groupby("Tecnica"):
        ax.scatter(
            grupo["PC1"],
            grupo["PC2"],
            grupo["PC3"],
            c=colores_tecnica.get(tecnica, "gray"),
            label=tecnica,
            s=60,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({varianza[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({varianza[1]*100:.0f}%)")
    ax.set_zlabel(f"PC3 ({varianza[2]*100:.0f}%)")
    ax.set_title("Clusterización de Técnicas mediante PCA", fontsize=14)
    ax.legend(title="Técnica")

    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Gráfico PCA guardado en: {filename}")
    plt.close()


def generate_pca_per_string(df, filename="guitarras_pca_cuerdas.png"):

    print("Generando PCA 3D por número de cuerda...")

    # Función para extraer el número de cuerda del nombre
    def detectar_cuerda(nombre_archivo):
        import re

        match = re.search(r"nota[-_]?(\d+)", nombre_archivo, re.IGNORECASE)
        if match:
            return int(
                match.group(1)
            )  # Devolvemos como ENTERO para que ordene bien (1, 2, 3...)
        return None

    df_pca = df.copy()
    df_pca["Cuerda"] = df_pca["Archivo"].apply(detectar_cuerda)

    # Eliminamos archivos donde no se detectó número
    df_pca = df_pca.dropna(subset=["Cuerda"])
    df_pca = df_pca.sort_values(by="Cuerda")

    cols_features = df_pca.select_dtypes(include=[np.number]).columns.tolist()
    # Quitamos la columna 'Cuerda' si se coló como numérica
    if "Cuerda" in cols_features:
        cols_features.remove("Cuerda")

    X = df_pca[cols_features].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=3)
    componentes = pca.fit_transform(X_scaled)

    df_pca[["PC1", "PC2", "PC3"]] = componentes
    varianza = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # para ver la transición de agudos (1) a graves (6)
    cuerdas_unicas = sorted(df_pca["Cuerda"].unique())
    cmap = plt.get_cmap("plasma", len(cuerdas_unicas))

    for i, cuerda in enumerate(cuerdas_unicas):
        grupo = df_pca[df_pca["Cuerda"] == cuerda]
        ax.scatter(
            grupo["PC1"],
            grupo["PC2"],
            grupo["PC3"],
            color=cmap(i),
            label=f"Cuerda {cuerda}",
            s=80,
            alpha=0.8,
            edgecolor="k",
            linewidth=0.5,
        )

    # Etiquetas
    ax.set_xlabel(f"PC1 ({varianza[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({varianza[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({varianza[2]*100:.1f}%)")
    ax.set_title("Distribución de características por cuerda (Frecuencia)", fontsize=14)

    # Leyenda
    ax.legend(title="Nº Cuerda", loc="upper left", bbox_to_anchor=(1.05, 1))

    ax.view_init(elev=30, azim=120)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Guardado: {filename}")
    plt.close()


def generate_correlation_matrix(df, filename="matriz_correlacion.png"):
    print("Generando Matriz de Correlación...")

    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    if (
        "Cuerda" in cols_numericas
    ):  # eliminamos columnas que no aporten informacion numerica relevante
        cols_numericas.remove("Cuerda")

    # Calculamos la matriz de correlación de Pearson
    corr = df[cols_numericas].corr()

    plt.figure(figsize=(10, 8))

    # Dibujamos el mapa de calor
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
        annot=True,
        fmt=".2f",
    )

    plt.title("Matriz de correlación de parámetros acústicos", size=15, pad=22)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Guardado: {filename}")
    plt.close()


def generate_small_multiples_bars(
    df, filename="graficos_small_multiples.png"
):  # datos globales por archivo, no por nota

    cols_a_excluir = [
        "Atk(s)",
        "Dec(s)",
        "Sus(s)",
        "Inharm",
        "Brillo (Nota)",
        "L/M (Nota)",
        "Numero_Nota",
        "Cuerda",
        "Grupo",
    ]

    # Identificamos columnas numéricas y filtramos
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_numericas = [col for col in cols_numericas if col not in cols_a_excluir]

    df_agrupado = df.groupby(["Grupo", "Clase"])[cols_numericas].mean().reset_index()

    df_norm = df_agrupado.copy()
    for col in cols_numericas:
        max_val = df_norm[col].max()
        min_val = df_norm[col].min()
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0.0

    def cambio_nombre_guitarra(nombre_grupo):
        match = re.search(r"g(\d+)", nombre_grupo, re.IGNORECASE)
        if match:
            return f"Guitarra {match.group(1)}"
        return nombre_grupo

    df_norm["Grupo"] = df_norm["Grupo"].apply(cambio_nombre_guitarra)

    df_melt = df_norm.melt(
        id_vars=["Grupo", "Clase"],
        value_vars=cols_numericas,
        var_name="Característica",
        value_name="Magnitud Normalizada",
    )

    # col_wrap=3 significa que pondrá 3 gráficos por fila
    g = sns.catplot(
        data=df_melt,
        x="Magnitud Normalizada",
        y="Grupo",
        col="Característica",
        hue="Clase",  # distingue colores por clase
        kind="bar",
        col_wrap=3,
        height=3.5,
        aspect=1.5,
        palette="Set2",
        sharex=True,
        legend=True,
    )

    g.set_titles(col_template="{col_name}", size=18, fontweight="bold")
    g.set_axis_labels("Magnitud normalizada (0-1)", "", fontsize=18)

    for ax in g.axes.flat:
        ax.tick_params(axis="y", labelsize=18)  # Nombres de las guitarras
        ax.tick_params(axis="x", labelsize=18)  # Números del eje X

    # Forzar la creación de la leyenda abajo usando la figura global
    handles, labels = g.axes[0].get_legend_handles_labels()

    if not handles and g._legend is not None:
        handles = g._legend.legend_handles
        labels = [text.get_text() for text in g._legend.get_texts()]

    if g._legend is not None:
        g._legend.remove()

    # Dibujamos la nuestra HORIZONTAL, abajo del todo y forzando los elementos capturados
    g.figure.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.02,
        ),  # Bajamos un pelín la coordenada para que no pise el eje X de abajo
        ncol=(
            len(labels) if labels else 2
        ),  # Se adapta dinámicamente al número de clases de forma horizontal
        fontsize=18,
        frameon=True,  # Le mete un recuadro muy fino de fondo para que destaque limpiamente
    )

    # Espaciado del título con un par de saltos de línea para alejarlo
    plt.subplots_adjust(
        top=0.85, hspace=0.45, bottom=0.15
    )  # Damos margen abajo para que quepa la leyenda
    # g.figure.suptitle(
    #     "Comparativa de características acústicas por guitarra e intérprete\n\n",
    #     fontsize=18,
    #     fontweight="bold",
    # )

    g.figure.suptitle(
        "Comparativa de características acústicas\n\n",
        fontsize=20,
        fontweight="bold",
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Guardado: {filename}")

    plt.close()


def graph_notes(df, filename="graficos_evolucion_notas.png"):

    # Filtrar solo las filas que corresponden a notas individuales
    df_notas = df[df["Archivo"].str.contains("-nota", case=False, na=False)].copy()

    if df_notas.empty:
        print("No se encontraron notas individuales para graficar la evolución.")
        return

    # extraer el número de nota
    df_notas["Numero_Nota"] = df_notas["Archivo"].str.extract(r"-nota(\d+)").astype(int)

    df_notas["Grupo"] = df_notas["Archivo"].apply(limpiar_nombre)

    df_notas["Guitarra"] = df_notas["Grupo"].apply(lambda x: str(x).split("-")[0])
    df_notas["Interprete_Nombre"] = df_notas["Grupo"].apply(
        lambda x: str(x).split("-")[1] if "-" in str(x) else str(x)
    )

    interpretes_unicos = sorted(df_notas["Interprete_Nombre"].unique())
    int_1 = interpretes_unicos[0]

    df_notas["Eje_X"] = df_notas.apply(
        lambda row: (
            row["Numero_Nota"]
            if row["Interprete_Nombre"] == int_1
            else row["Numero_Nota"] + 5
        ),
        axis=1,
    )

    cols_notas = ["Atk(s)", "Dec(s)", "Sus(s)", "Inharm", "Brillo (Nota)", "L/M (Nota)"]

    # Verificar que las columnas existan por si acaso
    cols_graficar = [c for c in cols_notas if c in df_notas.columns]

    # Transformar los datos a formato largo (melt) para Seaborn
    df_melt = df_notas.melt(
        id_vars=[
            "Eje_X",
            "Guitarra",
            "Grupo",
            "Interprete_Nombre",
        ],
        value_vars=cols_graficar,
        var_name="Característica",
        value_name="Valor",
    )

    # Generar cuadricula de gráficos
    g = sns.relplot(
        data=df_melt,
        x="Eje_X",
        y="Valor",
        col="Característica",
        hue="Guitarra",
        units="Grupo",
        estimator=None,
        kind="line",
        palette="Set2",
        col_wrap=2,  # num graficos por fila
        height=3,
        aspect=1.5,
        marker="o",
        facet_kws={"sharey": False},
    )

    g.set_titles(col_template="{col_name}", size=14, fontweight="bold")

    # Lo quitamos del ajuste general para controlarlo nosotros mismos gráfico por gráfico
    g.set_axis_labels("", "Valor")

    for ax in g.axes.flat:
        ax.set_xticks(range(1, 11))
        # cuerdas al aire
        etiquetas = [
            "5\n(La2/La3)",
            "4\n(Re3/Re4)",
            "3\n(Sol3)",
            "2\n(Si3)",
            "1\n(Mi4)",
            "5\n(La2/La3)",
            "4\n(Re3/Re4)",
            "3\n(Sol3)",
            "2\n(Si3)",
            "1\n(Mi4)",
        ]
        ax.set_xticklabels(etiquetas)

        ax.set_xlabel("Orden (Nota al aire)", labelpad=22)

        ax.axvline(x=5.5, color="gray", linestyle=":", alpha=0.5)

        nombre_1 = "Intérprete 1"  # alejandro
        ax.text(
            0.25,
            -0.14,
            nombre_1,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            color="gray",
            fontweight="bold",
        )

        if len(interpretes_unicos) > 1:
            nombre_2 = "Intérprete 2"  # uxia
            ax.text(
                0.75,
                -0.14,
                nombre_2,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
                color="gray",
                fontweight="bold",
            )

        ax.grid(True, linestyle="--", alpha=0.4)

    # hspace a 0.4 para que las filas tengan espacio de sobra para este eje X más alto
    plt.subplots_adjust(top=0.85, hspace=0.3)
    g.figure.suptitle(
        "Evolución temporal de características por nota", fontsize=18, fontweight="bold"
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Guardado: {filename}")
    plt.close()
