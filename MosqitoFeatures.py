import mosqito
import numpy as np
from scipy.stats import kurtosis

from mosqito.sound_level_meter import comp_spectrum
from mosqito.sound_level_meter import noct_spectrum
from mosqito.sound_level_meter import noct_synthesis

from mosqito.sq_metrics import loudness_ecma
from mosqito.sq_metrics import loudness_zwst
from mosqito.sq_metrics import loudness_zwst_freq
from mosqito.sq_metrics import loudness_zwst_perseg
from mosqito.sq_metrics import loudness_zwtv

from mosqito.sq_metrics import sharpness_din_st
from mosqito.sq_metrics import sharpness_din_tv 
from mosqito.sq_metrics import sharpness_din_freq
from mosqito.sq_metrics import sharpness_din_perseg
from mosqito.sq_metrics import sharpness_din_from_loudness

from mosqito.sq_metrics import roughness_dw
from mosqito.sq_metrics import roughness_dw_freq
from mosqito.sq_metrics import roughness_ecma

from mosqito.sq_metrics import pr_ecma_st
from mosqito.sq_metrics import pr_ecma_freq
from mosqito.sq_metrics import pr_ecma_perseg
from mosqito.sq_metrics import tnr_ecma_st
from mosqito.sq_metrics import tnr_ecma_freq
from mosqito.sq_metrics import tnr_ecma_perseg

from mosqito.sq_metrics import sii_ansi
from mosqito.sq_metrics import sii_ansi_freq
from mosqito.sq_metrics import sii_ansi_level

from mosqito.utils import amp2db
from mosqito.utils import db2amp
from mosqito.utils import bark2freq
from mosqito.utils import freq2bark
from mosqito.utils import spectrum2dBA

from mosqito.utils import sine_wave_generator
from mosqito.utils import am_sine_generator
from mosqito.utils import am_noise_generator
from mosqito.utils import fm_sine_generator

from mosqito.utils import time_segmentation

from mosqito.utils import LTQ

from mosqito.utils import load

print(mosqito.__version__)



# hay que definir una clase en la que defina todos los métodos:


class MosqitoFeatures:
    def __init__(self, signal, fs):
        self.signal = signal
        self.fs = fs

# FUNCION PARA DIVIDIR POR TRAMAS 
    def frame_analysis(self, feature_name, frame_duration=0.1):
        """Análisis por tramas para características básicas"""
        frame_len = int(frame_duration * self.fs)
        num_frames = len(self.signal) // frame_len

        features = []
        times = []

        for i in range(num_frames):
            start = i * frame_len
            end = start + frame_len
            frame = self.signal[start:end]

            # Normalizar solo si no es energía
            if feature_name != "energy":
                std = np.std(frame) + 1e-10
                frame = frame / std
                frame = frame - np.mean(frame)

            if feature_name == "centroid":
                X = np.abs(np.fft.fft(frame, n=frame_len))
                f = np.fft.fftfreq(frame_len, d=1 / self.fs)
                n = frame_len // 2
                X, f = X[:n], f[:n]
                value = np.sum(f * X) / (np.sum(X) + 1e-10)

            elif feature_name == "zcr":
                value = np.sum(np.abs(np.diff(np.sign(frame)))) / frame_len

            elif feature_name == "energy":
                value = np.sum(frame ** 2) / frame_len

            elif feature_name == "kurtosis":
                value = kurtosis(frame)

            else:
                raise ValueError(f"Característica desconocida: {feature_name}")


            features.append(value)
            times.append(start / self.fs)

        return np.array(times), np.array(features)
    
    # Loudness
    def get_loudness_ecma(self):
        """Cálculo de Loudness ECMA"""
        return loudness_ecma(self.signal, self.fs)

    def get_loudness_zwst(self):
        """Cálculo de Loudness Stationary (ISO 532-1:2017)"""
        return loudness_zwst(self.signal, self.fs)

    def get_loudness_zwst_freq(self):
        """Cálculo de Loudness Stationary con frecuencia"""
        return loudness_zwst_freq(self.signal, self.fs)

    def get_loudness_zwst_perseg(self):
        """Cálculo de Loudness Stationary con segmentos"""
        return loudness_zwst_perseg(self.signal, self.fs)

    def get_loudness_zwtv(self):
        """Cálculo de Loudness Time-varying (ISO 532-1:2017)"""
        return loudness_zwtv(self.signal, self.fs)

     # Sharpness
    def get_sharpness_din_st(self):
        return sharpness_din_st(self.signal, self.fs)

    def get_sharpness_din_tv(self):
        return sharpness_din_tv(self.signal, self.fs)

    def get_sharpness_din_freq(self):
        return sharpness_din_freq(self.signal, self.fs)

    def get_sharpness_din_perseg(self):
        return sharpness_din_perseg(self.signal, self.fs)

    def get_sharpness_din_from_loudness(self):
        spec, _ = loudness_zwst_freq(self.signal, self.fs)
        return sharpness_din_from_loudness(spec)

    # Roughness
    def get_roughness_dw(self):
        return roughness_dw(self.signal, self.fs)

    def get_roughness_dw_freq(self):
        return roughness_dw_freq(self.signal, self.fs)

    def get_roughness_ecma(self):
        return roughness_ecma(self.signal, self.fs)

    # Tonality
    def get_pr_ecma_st(self):
        """Cálculo de Tonality Prominence Ratio"""
        return pr_ecma_st(self.signal, self.fs)

    def get_pr_ecma_freq(self):
        """Cálculo de Tonality Prominence Ratio"""
        return pr_ecma_freq(self.signal, self.fs)

    def get_pr_ecma_perseg(self):
        """Cálculo de Tonality Prominence Ratio"""
        return pr_ecma_perseg(self.signal, self.fs)

    def get_tnr_ecma_st(self):
        """Cálculo de Tonality Tone-to-noise Ratio"""
        return tnr_ecma_st(self.signal, self.fs)

    def get_tnr_ecma_freq(self):
        """Cálculo de Tonality Tone-to-noise Ratio"""
        return tnr_ecma_freq(self.signal, self.fs)

    def get_tnr_ecma_perseg(self):
        """Cálculo de Tonality Tone-to-noise Ratio"""
        return tnr_ecma_perseg(self.signal, self.fs)

    # Speech Intelligibility Index
    def get_sii_ansi(self):
        return sii_ansi(self.signal, self.fs)

    def get_sii_ansi_freq(self):
        return sii_ansi_freq(self.signal, self.fs)

    def get_sii_ansi_level(self):
        return sii_ansi_level(self.signal, self.fs)
    
    
    
    
        # Conversión y utilidades varias

    def get_amp2db(self):
        """Conversión de amplitud a decibelios"""
        return amp2db(self.signal)

    def get_db2amp(self):
        """Conversión de decibelios a amplitud"""
        return db2amp(self.signal)

    def get_bark2freq(self, bark_vals):
        """Conversión de Bark a Frecuencia"""
        return bark2freq(bark_vals)

    def get_freq2bark(self, freq_vals):
        """Conversión de Frecuencia a Bark"""
        return freq2bark(freq_vals)

    def get_spectrum2dBA(self):
        """Conversión de espectro a dBA"""
        spectrum, _ = comp_spectrum(self.signal, self.fs)
        return spectrum2dBA(spectrum)
    
    def get_noct_spectrum(self):
        """Cálculo del espectro de bandas de tercio de octava (nocturnal spectrum)"""
        return noct_spectrum(self.signal, self.fs)

    def get_noct_synthesis(self):
        """Síntesis de señal a partir de un espectro de bandas de tercio de octava"""
        spectrum, _ = noct_spectrum(self.signal, self.fs)
        return noct_synthesis(spectrum, self.fs)


    # Generadores de señal

    def generate_sine_wave(self, freq=1000, duration=1.0, rms=0.1):
        """Generador de onda seno"""
        return sine_wave_generator(freq, self.fs, duration, rms)

    def generate_am_sine(self, carrier=1000, modulator=20, duration=1.0, rms=0.1):
        """Generador de onda seno AM"""
        return am_sine_generator(carrier, modulator, self.fs, duration, rms)

    def generate_am_noise(self, mod_freq=4, duration=1.0, rms=0.1):
        """Generador de ruido AM"""
        return am_noise_generator(mod_freq, self.fs, duration, rms)

    def generate_fm_sine(self, carrier=1000, modulator=20, duration=1.0, rms=0.1):
        """Generador de onda seno FM"""
        return fm_sine_generator(carrier, modulator, self.fs, duration, rms)

    # Segmentación temporal

    def get_time_segmentation(self, seg_size=1024, overlap=0):
        """Segmentación temporal de la señal"""
        return time_segmentation(self.signal, self.fs, seg_size, overlap)

    # Carga de archivos

    def load_signal(self, file_path):
        """Carga de señal desde archivo"""
        self.signal, self.fs = load(file_path)

    # Umbral de audición (LTQ)

    def get_ltq(self):
        """Retorno del umbral de audición LTQ"""
        return LTQ
   
    # Generadores de señal

    def generate_sine_wave(self, freq=1000, duration=1.0, rms=0.1):
        """Generador de onda seno"""
        return sine_wave_generator(freq, self.fs, duration, rms)

    def generate_am_sine(self, carrier=1000, modulator=20, duration=1.0, rms=0.1):
        """Generador de onda seno AM"""
        return am_sine_generator(carrier, modulator, self.fs, duration, rms)

    def generate_am_noise(self, mod_freq=4, duration=1.0, rms=0.1):
        """Generador de ruido AM"""
        return am_noise_generator(mod_freq, self.fs, duration, rms)

    def generate_fm_sine(self, carrier=1000, modulator=20, duration=1.0, rms=0.1):
        """Generador de onda seno FM"""
        return fm_sine_generator(carrier, modulator, self.fs, duration, rms)

    # Segmentación temporal

    def get_time_segmentation(self, seg_size=1024, overlap=0):
        """Segmentación temporal de la señal"""
        return time_segmentation(self.signal, self.fs, seg_size, overlap)

    # Carga de archivos

    def load_signal(self, file_path):
        """Carga de señal desde archivo"""
        self.signal, self.fs = load(file_path)

    # Umbral de audición (LTQ)

    def get_ltq(self):
        """Retorno del umbral de audición LTQ"""
        return LTQ

  
