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
WINDOW_STEP = 21050
NUM_HPS = 5
POWER_THRESH = 1e-6
CONCERT_PITCH = 440
# se calcula con una nota de referencia, como La a 440 Hz
# https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html

HANN_WINDOW = np.hanning(WINDOW_SIZE)
WHITE_NOISE_THRESH = 0.2
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

audio_file = "audio/MIS_AUDIOS/Notas con pua, electrica.m4a"
# ____________________________________________________________________


def find_closest_note(pitch):
    i = int(np.round(np.log2(pitch / CONCERT_PITCH) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = CONCERT_PITCH * 2 ** (i / 12)
    return closest_note, closest_pitch


def process_window(window_samples):
    # skip if signal power is too low
    signal_power = (np.linalg.norm(window_samples, ord=2) ** 2) / len(window_samples)
    if signal_power < POWER_THRESH:
        print("Closest note: ...")
        return

    # avoid spectral leakage
    hann_samples = window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[: len(hann_samples) // 2])
    DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE

    # supress mains hum
    for i in range(int(62 / DELTA_FREQ)):
        magnitude_spec[i] = 0

    # suppress low energy frequencies per octave band
    for j in range(len(OCTAVE_BANDS) - 1):
        ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
        ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
        ind_end = min(ind_end, len(magnitude_spec))
        avg_energy_per_freq = (
            np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2) ** 2
            / (ind_end - ind_start)
        ) ** 0.5
        for i in range(ind_start, ind_end):
            if magnitude_spec[i] < WHITE_NOISE_THRESH * avg_energy_per_freq:
                magnitude_spec[i] = 0

    # interpolate spectrum
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

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    print(f"Closest note: {closest_note} {max_freq}/{closest_pitch}")


def main():
    if not os.path.isfile(audio_file):
        print(f"Audio file {audio_file} not found!")
        return
    data = sf.read(audio_file)
    if len(data.shape) > 1:
        data = data[:, 0]  # tomar solo un canal si es stereo

    window_samples = np.zeros(WINDOW_SIZE)

    for start in range(0, len(data), WINDOW_STEP):
        end = start + WINDOW_STEP
        window_samples = np.roll(window_samples, -WINDOW_STEP)
        window_samples[-WINDOW_STEP:] = (
            data[start:end]
            if end <= len(data)
            else np.pad(data[start:], (0, WINDOW_STEP - (len(data) - start)))
        )
        process_window(window_samples)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
