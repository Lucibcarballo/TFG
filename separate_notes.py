# CODIGO PARA SEPARAR NOTAS EN AUDIOS CON VARIAS NOTAS, FORZANDO LA DETECCIÓN DE 6 NOTAS POR AUDIO

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile as sf
import os
import librosa

# _____________________configuracion___________________________
base_path = "C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/"
output_path = "C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/notas_separadas/"
duration_seconds = 1.5  # duración de cada segmento en segundos
n_notes = 6  # número de notas a detectar por archivo
# _____________________________________________________________


def load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"El path {path} no existe.")

    y, fs = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Normalización
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, fs  #


def get_forced_onsets(y, fs, n_notes):
    # forzar detección de n notas, cogiendo los picos mas altos y asegurando distancia mínima entre ellos

    # Calculamos la fuerza de los ataques
    onset_env = librosa.onset.onset_strength(y=y, sr=fs)

    # Obtenemos los picos locales
    window = int(0.1 * fs / 512)  # ventana de comparación (aprox 0.1s)
    peaks = librosa.util.peak_pick(
        onset_env,
        pre_max=window,
        post_max=window,
        pre_avg=window,
        post_avg=window,
        delta=0.01,
        wait=1,
    )  # wait=1 para cogerlo todo

    # Obtenemos la fuerza de cada pico detectado
    peak_strengths = onset_env[peaks]

    # Creamos una lista de (indice_frame, fuerza)
    candidates = list(zip(peaks, peak_strengths))

    # Ordenamos por fuerza (del más fuerte al más debil) para coger solo los 6 que buscamos
    candidates.sort(key=lambda x: x[1], reverse=True)

    final_onsets = []
    # Distancia mínima entre notas en frames (aprox 0.25 segundos)
    min_dist_frames = int(0.25 * fs / 512)

    # Selección iterativa:
    for frame, strength in candidates:
        if len(final_onsets) >= n_notes:
            break

        is_far_enough = True
        for existing_frame in final_onsets:
            if abs(frame - existing_frame) < min_dist_frames:
                is_far_enough = False
                break

        if is_far_enough:
            final_onsets.append(frame)

    # Reordenar cronológicamente
    final_onsets.sort()

    # BACKTRACK: Ajustar el corte al momento exacto donde empieza a subir la onda
    final_onsets_backtracked = librosa.onset.onset_backtrack(
        np.array(final_onsets), onset_env
    )

    return librosa.frames_to_samples(final_onsets_backtracked)


def plot_segmentation(base_path, audio_file):
    full_path = os.path.join(base_path, audio_file)
    y, fs = load_audio(full_path)
    if y is None:
        return

    onset_samples = get_forced_onsets(y, fs, n_notes=6)

    onset_times = librosa.samples_to_time(onset_samples, sr=fs)

    plt.figure(figsize=(14, 4))
    librosa.display.waveshow(y, sr=fs, alpha=0.6)

    plt.vlines(
        onset_times, -1, 1, color="r", linewidth=2, linestyle="--", label="Corte"
    )

    plt.title(f"{audio_file} - Detectadas: {len(onset_samples)} notas")
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_segments(base_path, output_path, audio_file, duration_seconds, n_notes):
    full_path = os.path.join(base_path, audio_file)
    y, fs = load_audio(full_path)

    onsets = get_forced_onsets(y, fs, n_notes)
    segment_length = int(duration_seconds * fs)

    clean_name = audio_file.replace(".wav", "")

    print(f"--> Procesando y guardando notas de: {audio_file}")

    for i, start in enumerate(onsets):
        end = start + segment_length

        # Padding (rellenar con ceros) si el segmento se sale de la longitud del audio
        # para asegurar que todos duren exactamente lo mismo (mejor para luego poder comparar bien)
        if end > len(y):
            padding = np.zeros(end - len(y), dtype=np.float32)
            y_segment = np.concatenate((y[start:], padding))
        else:
            y_segment = y[start:end]

        out_name = f"{clean_name}_nota_{i+1}.wav"
        out_file = os.path.join(output_path, out_name)

        sf.write(out_file, y_segment, fs)


def main():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mis_archivos = [
        "notas_pua_electrica.wav",
        "notas_pua_española.wav",
        "notas_uña_electrica.wav",
        "notas_uña_española.wav",
        "notas_yema_electrica.wav",
        "notas_yema_española.wav",
    ]

    for archivo in mis_archivos:
        plot_segmentation(base_path, archivo)
        save_segments(base_path, output_path, archivo, duration_seconds, n_notes)

    print("Proceso completado. Notas guardadas en:", output_path)


if __name__ == "__main__":
    main()
