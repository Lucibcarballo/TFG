import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import os


def load_audio(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"El path {path} no existe.")

    y, fs = sf.read(path)

    # Si el audio es estereo, convertir a mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    if y.dtype != np.float32:
        y = y / np.max(np.abs(y))  # normalizar

    return y, fs


def segment_by_onsets(y, fs, duration_seconds=1.5):  # para detectar inicios de notas

    # backtrack=True ajusta el inicio al momento exacto del ataque
    onsets = librosa.onset.onset_detect(y=y, sr=fs, backtrack=True, units="samples")

    segment_length = int(duration_seconds * fs)

    for start_sample in onsets:
        end_sample = start_sample + segment_length

        if end_sample > len(y):
            continue

        y_segment = y[start_sample:end_sample]

    return y_segment


def main():
    audio_path = f"C:/Users/lucib/Desktop/TFG/audio/MIS_AUDIOS/notas_pua_electrica.wav"
    y, fs = load_audio(audio_path)

    mis_archivos = [
        {"path": "notas_pua_electrica.wav", "clase": "electrica"},
        {"path": "notas_pua_española.wav", "clase": "española"},
        {"path": "notas_uña_electrica.wav", "clase": "electrica"},
        {"path": "notas_uña_española.wav", "clase": "española"},
        {"path": "notas_yema_electrica.wav", "clase": "electrica"},
        {"path": "notas_yema_española.wav", "clase": "española"},
    ]

    dataset = []
    for item in mis_archivos:
        y, fs = load_audio(item["path"])
        # Sacamos múltiples muestras de cada audio
        segmentos_feat = segment_by_onsets(y, fs)

        for f in segmentos_feat:
            f["label"] = item["clase"]  # Etiqueta: española o electrica
            dataset.append(f)

    df = pd.DataFrame(dataset)


if __name__ == "__main__":
    main()
