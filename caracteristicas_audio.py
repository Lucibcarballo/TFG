import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import get_window, resample_poly, spectrogram, find_peaks
from MosqitoFeatures import MosqitoFeatures
import mosqito as mq
import numpy as np
import librosa
import os

# def load_audio(path):
#     y, fs = sf.read(path)

#     # Si el audio es estereo, convertir a mono
#     if y.ndim > 1:
#         y = y.mean(axis=1)

#     if y.dtype != np.float32:
#         y = y / np.max(np.abs(y))  # normalizar

#     return y, fs


def load_audio_package(folder_path):
    folder_path = os.path.abspath(folder_path)

    # Buscar wav con "c12" en el nombre
    wav_c12 = None
    for f in os.listdir(folder_path):
        if f.lower().endswith(".wav") and "c12" in f.lower():
            wav_c12 = os.path.join(folder_path, f)
            break
    if wav_c12 is None:
        raise FileNotFoundError(
            "No se encontró ningún .wav que contenga 'c12' en el nombre."
        )

    # Buscar info.txt
    info_path = os.path.join(folder_path, "info.yaml")
    if not os.path.exists(info_path):
        raise FileNotFoundError("No se encontró 'info.yaml' en la carpeta.")

    # Leer el WAV
    audio, fs = sf.read(wav_c12)

    # Leer info.txt
    info = {}
    with open(info_path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                try:
                    value = float(value)
                except:
                    pass
                info[key] = value

    return audio, fs, info


def ADSR_curve(y, fs, n_fft=2048, hop_length=256):
    # nfft = tamaño ventana, hop_length= cuanto se desplaza la ventana

    S = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    )  # stft: short fourier transform
    env = np.max(S, axis=0)  # máximo por frame
    env = env / np.max(env)  # normalizar

    # Encontrar índices aproximados de ADSR
    # ataque: tiempo hasta alcanzar 90% de la amplitud máxima
    attack_idx = np.argmax(env >= 0.9)

    # decay: tiempo hasta el nivel de sustain (ej. 50%)
    decay_idx = attack_idx + np.argmax(env[attack_idx:] <= 0.5)

    # sustain: período hasta que comienza la caída final
    sustain_idx = decay_idx + np.argmax(env[decay_idx:] <= 0.5)  # ejemplo simplificado

    # release: desde el final del sustain hasta que env≈0
    release_idx = len(env) - 1  # asumimos que termina al final del clip

    # Convertir frames a tiempo
    times = np.arange(len(env)) * hop_length / fs
    attack_time = times[attack_idx]
    decay_time = times[decay_idx]
    sustain_time = times[sustain_idx]
    release_time = times[release_idx]

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
    n_samples = len(y)  # num muestras
    fft_vals = np.fft.rfft(y)  # transformada rapida de fourier
    freqs = np.fft.rfftfreq(n_samples, 1 / fs)
    mag = np.abs(fft_vals)

    return freqs, mag


def compute_spectrogram(y, fs, nperseg=1024, noverlap=512):
    f, t, Sxx = spectrogram(y, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)

    energy_pulse = np.sum(
        Sxx, axis=0
    )  # suma energía en todas las frecuencias para cada instante

    eps = 1e-10  # para evitar log(0)
    energy_db = 10 * np.log10(energy_pulse + eps)

    plt.figure(figsize=(10, 4))
    plt.plot(t, energy_db, color="purple")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Energía total [dB]")
    plt.title("Pulso temporal de energía (suma de todas las frecuencias)")
    plt.grid(True)
    plt.show()

    # Promedio por filas (energía promedio de cada frecuencia)
    avg_spectrum = np.mean(Sxx, axis=1)
    avg_db = 10 * np.log10(avg_spectrum + 1e-10)

    peaks, _ = find_peaks(
        avg_db,
        prominence=2,  # pico debe sobresalir al menos 3 dB del entorno
        height=np.max(avg_db) - 40,  # evita picos pequeños
        distance=10,  # evita picos pegados
    )

    plt.figure(figsize=(10, 4))
    plt.plot(f, avg_db, color="blue")
    plt.plot(
        f[peaks], avg_db[peaks], "ro", markersize=5
    )  # puntos rojos para marcar armonicos
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Energía promedio [dB]")
    plt.title("Promedio espectral (sobre tiempo) → armónicos")
    plt.grid(True)
    plt.show()

    print("\nFrecuencias de los picos detectados:")
    for fp in f[peaks]:
        print(f" {fp:.2f} Hz")

    return f, t, Sxx, peaks


import numpy as np


def compute_inharmonicity(
    peaks, f
):  # la inarmonicidad (B) describe cuánto se desvían los armónicos reales de un sonido respecto de los armónicos ideales

    fp = f[peaks]  # frecuencias de los picos
    Num = len(fp)  # número de armónicos que consideramos
    Fundamental = fp[0]  # asumimos el primer pico como fundamental
    armonicos = np.arange(1, Num + 1)

    B = ((fp[:Num] / (armonicos * Fundamental)) ** 2 - 1) / (armonicos**2)
    print("Inarmonicidad:", B)

    return B


def subband_energy(y, fs, N, freqs, mag):
    f_min, f_max = freqs.min(), freqs.max()
    cutoffs = np.linspace(
        f_min, f_max, N + 1
    )  # N bandas → N+1 puntos (incluye extremos)

    # integrar energía en cada banda
    band_energies = []
    for i in range(N):
        band_mask = (freqs >= cutoffs[i]) & (freqs < cutoffs[i + 1])
        band_energy = np.sum(mag[band_mask])
        band_energies.append(band_energy)

    # visualizacion de bandas de frecuencia
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mag)
    for c in cutoffs:
        plt.axvline(c, color="r", linestyle="--", alpha=0.5)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Energía promedio")
    plt.title("División en bandas de frecuencia")
    plt.show()

    # ________________________________________________Pulso de energía por subbanda
    frame_len = int(0.05 * fs)  # 50 ms
    hop = int(0.025 * fs)  # 25 ms solapamiento
    window = get_window("hann", frame_len)

    bands = {"graves": (20, 200), "medios": (200, 2000), "agudos": (2000, fs / 2 - 100)}

    n_frames = int((len(y) - frame_len) / hop) + 1
    subband_energy = {name: np.zeros(n_frames) for name in bands}

    for i in range(n_frames):
        frame = y[i * hop : i * hop + frame_len] * window
        fft_vals = np.fft.rfft(frame)
        freqs = np.fft.rfftfreq(frame_len, 1 / fs)
        mag2 = np.abs(fft_vals) ** 2  # energía

        for name, (f_low, f_high) in bands.items():
            mask = (freqs >= f_low) & (freqs < f_high)
            subband_energy[name][i] = np.sum(mag2[mask])

    t_frames = np.arange(n_frames) * hop / fs

    plt.figure(figsize=(10, 6))

    for i, (name, energy) in enumerate(subband_energy.items(), 1):
        plt.subplot(len(subband_energy), 1, i)
        energy_norm = energy / np.max(energy)  # opcional para normalizar entre 0-1
        plt.plot(t_frames, energy_norm, color="C" + str(i))
        plt.title(f"Subbanda {name}")
        plt.ylabel("Energía normalizada")
        plt.grid(True)

    plt.xlabel("Tiempo [s]")
    plt.tight_layout()
    plt.show()

    return band_energies


def mosqito_features(y, fs):
    mf = MosqitoFeatures(y, fs)

    features_frame = {
        "centroid": mf.frame_analysis("centroid"),
        "zcr": mf.frame_analysis("zcr"),
        "energy": mf.frame_analysis("energy"),
        "kurtosis": mf.frame_analysis("kurtosis"),
    }

    return features_frame


def extract_mosqito_features(y, fs):
    # _______________________________________________Extraer caracteristicas del audio completo y representarlas

    features_frame = mosqito_features(y, fs)

    # LOUDNESS
    plt.figure(figsize=(10, 6))
    N, N_spec, bark_axis = mq.loudness_zwst(y, fs)
    print("Loudness:", N)
    plt.plot(bark_axis, N_spec)
    plt.xlabel("Frequency band [Bark]")
    plt.ylabel("Specific loudness [Sone/Bark]")
    plt.title("Loudness = " + f"{N:.2f}" + " [Sone]")

    # SHARPNESS
    S = mq.sharpness_din_from_loudness(N, N_spec)  # S en [acum]
    print("Sharpness (from loudness):", S)

    y_48k = resample_poly(y, 48000, fs)  # hay que muestrear a 48k para que lo acepte

    S_tv, t_tv = mq.sharpness_din_tv(
        y_48k, 48000, skip=0.1
    )  # saltamos los primeros 0.1 s para evitar el efecto transitorio inicial
    plt.figure()
    plt.plot(t_tv, S_tv)
    plt.xlabel("Time [s]")
    plt.ylabel("Sharpness [acum]")
    plt.title("Sharpness across time")
    plt.grid(True)

    # ROUGHNESS
    R, R_specific, bark, time = mq.roughness_dw(y, fs, overlap=0)
    plt.figure()
    plt.plot(bark, R_specific)
    plt.xlabel("Bark axis [Bark]")
    plt.ylabel("Specific roughness [Asper/Bark]")
    plt.title("Roughness = " + f"{R[0]:.2f}" + " [Asper]")

    # TONALITY
    spectrum_db, freq_axis = mq.comp_spectrum(y, fs, db=True)
    t_pr, pr, prom, tones_freqs = mq.pr_ecma_st(y, fs)  # prominence ratio
    plt.figure()
    plt.bar(tones_freqs, pr, width=50)
    plt.grid(axis="y")
    plt.ylabel("PR [dB]")
    # plt.title("Total PR = "+ f"{t_pr[0]:.2f}" + " dB")
    plt.title("Total PR")
    print("Valor PR (en dB):", t_pr)
    plt.xscale("log")
    xticks_pos = list(tones_freqs) + [100, 1000, 10000]
    xticks_pos = np.sort(xticks_pos)
    xticks_label = [str(elem) for elem in xticks_pos]
    plt.xticks(xticks_pos, labels=xticks_label, rotation=30)
    plt.xlabel("Frequency [Hz]")

    t_tnr, tnr, prom, tones_freqs = mq.tnr_ecma_st(y, fs)  #  tone-to-noise ratio
    plt.figure()
    plt.bar(tones_freqs, tnr, width=50)
    plt.grid(axis="y")
    plt.ylabel("TNR [dB]")
    # plt.title("Total TNR = "+ f"{t_tnr[0]:.2f}" + " dB")
    plt.title("Total TNR")
    print("Valor TNR (en dB):", t_tnr)
    plt.xscale("log")
    xticks_pos = list(tones_freqs) + [100, 1000, 10000]
    xticks_pos = np.sort(xticks_pos)
    xticks_label = [str(elem) for elem in xticks_pos]
    plt.xticks(xticks_pos, labels=xticks_label, rotation=30)
    plt.xlabel("Frequency [Hz]")

    #### estas características salen a 0 si hay uniformidad de la señal

    # _______________________________________________Calculo promedio de cada característica
    features_mean = {
        name: np.nanmean(values)
        for name, (
            t,
            values,
        ) in features_frame.items()  # hacemos nanmean en vez de mean porque asi ignora los valores nan, si no kurtosis devuelve nan
    }

    # Muestra valores en terminal
    print("Valores promedio del audio:")
    for name, val in features_mean.items():
        print(f"{name.capitalize():<10}: {val:.4f}")


def main():
    audio_path = "audio/doviso"
    y, fs, info = load_audio_package(audio_path)

    print("Metadatos del archivo:", info)
    # aplicar calibración
    if "nivel" in info:
        escala = 10 ** (info["nivel"] / 20)
        y = y * escala

    adsr = ADSR_curve(y, fs)
    plot_adsr(adsr)

    freqs, mag = compute_FFT(y, fs)

    # _____________________________________________Representar energia vs frecuencia

    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, mag)  # escala logaritmica
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Energía (escala log)")
    plt.title("Energía en función de la frecuencia")
    plt.grid(True)
    plt.show()

    f, t, Sxx, peaks = compute_spectrogram(y, fs)
    B = compute_inharmonicity(peaks, f)

    N = 3  # bandas de frecuencia
    band_energies = subband_energy(y, fs, N, freqs, mag)

    brillantez = band_energies[2] / band_energies[1]  # indice brillantez
    print(f"Indice de brillantez: {brillantez:.4f}")

    IL = band_energies[0] / band_energies[1]  # indice de bajos a medios
    print(f"Indice de bajos a medios: {IL:.4f}")

    extract_mosqito_features(y, fs)

    print("Fin del programa.")


if __name__ == "__main__":
    main()
