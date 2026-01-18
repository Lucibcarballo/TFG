"""
Guitar tuner script based on the Harmonic Product Spectrum (HPS)

MIT License
Copyright (c) 2021 chciken

https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html
"""

# MODIFICADO PARA LEER DE UN AUDIO QUE LE PASAMOS EN VEZ DE DESDE EL MICROFONO

import copy
import os
import numpy as np
import scipy.fftpack
import soundfile as sf
import time

# ___________________________configuracion___________________________
SAMPLE_FREQ = 44100
WINDOW_SIZE = 44100
WINDOW_STEP = 44100
NUM_HPS = 3
POWER_THRESH = 1e-4
CONCERT_PITCH = 440
# se calcula con una nota de referencia, como La a 440 Hz
# https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html

HANN_WINDOW = np.hanning(WINDOW_SIZE)
WHITE_NOISE_THRESH = 0.2
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

audio_file = "C:\\Users\\lucib\\Desktop\\TFG\\audio\\MIS_AUDIOS\\notas_uña_española.wav"
# ____________________________________________________________________


def find_closest_note(pitch):
    if pitch <= 0:
        return "...", 0
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch


def process_window(window_samples):
    # skip if signal power is too low
    signal_power = (np.linalg.norm(window_samples, ord=2) ** 2) / len(window_samples)
    if signal_power < POWER_THRESH:
        return None

    # avoid spectral leakage
    hann_samples = window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[: len(hann_samples) // 2])
    DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE

    # supress mains hum
    for i in range(int(62 / DELTA_FREQ)):
        magnitude_spec[i] = 0

    # interpolate spectrum, algorithm HPS
    mag_spec_ipol = np.interp(
        np.arange(0, len(magnitude_spec), 1 / NUM_HPS),
        np.arange(0, len(magnitude_spec)),
        magnitude_spec,
    )
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)
    hps_spec = copy.deepcopy(mag_spec_ipol)

    # calculate HPS
    for i in range(NUM_HPS):
        tmp_hps_spec = np.multiply(
            hps_spec[: int(np.ceil(len(mag_spec_ipol) / (i + 1)))],
            mag_spec_ipol[:: (i + 1)],
        )
        if not any(tmp_hps_spec):
            break
        hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

    # FILTRO DE RANGO: Ignorar frecuencias fuera de una guitarra real (70Hz - 1000Hz)
    if max_freq < 70 or max_freq > 1000:
        return None

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    return closest_note, max_freq, closest_pitch


def main():
    if not os.path.isfile(audio_file):
        print(f"Audio file {audio_file} not found!")
        return

    data, fs = sf.read(audio_file)
    if len(data.shape) > 1:
        data = data[:, 0]  # tomar solo un canal si es stereo
    global SAMPLE_FREQ
    SAMPLE_FREQ = fs  # sincronizar con la frecuencia real del archivo

    last_note = ""

    for start in range(0, len(data) - WINDOW_SIZE, WINDOW_STEP):
        window = data[start : start + WINDOW_SIZE]
        result = process_window(window)

        if result:
            note, freq, pitch = result
            # FILTRO DE ESTABILIDAD: Solo imprimir si la nota cambia
            if note != last_note:
                print(f"Closest note: {note} Freq: {str(freq)}/{pitch} Hz")

                last_note = note
        else:
            # Si hay silencio, permitimos que se vuelva a detectar la misma nota después
            last_note = ""


if __name__ == "__main__":
    main()
