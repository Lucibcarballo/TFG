import pyplnoise
import numpy as np
import matplotlib.pyplot as plt

from MosqitoFeatures import MosqitoFeatures  


#_______________________Este codigo genera señales sinteticas: ruido rosa, ruido blanco y una señal tonal de 500 Hz_________________________

 
fs = 8000  
n = np.arange(16000)

# Ruido blanco

white = pyplnoise.WhiteNoise(fs, psd=1.0)
y1 = white.get_series(16000)

# Ruido rosa
pink = pyplnoise.PinkNoise(fs, 1e-3, 50.)

y2 = pink.get_series(32000)
y2 = y2 / np.std(y2) 

# Señal tonal de 500 Hz
y3 = np.sin(2 * np.pi * 500 / fs * n)

# Señal final
y = np.concatenate([y1, 20* y2[:16000], y3 + y1])

L = len(y)

tiempo = np.arange(L) / fs

# Visualización
plt.figure(figsize=(12, 4))
plt.plot(tiempo, y)
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal combinada: ruido blanco, rosa y tonal')
plt.grid(True)
plt.show()

# Análisis con MosqitoFeatures
mf = MosqitoFeatures(y, fs)

features = {
    "centroid": mf.frame_analysis("centroid"),
    "zcr": mf.frame_analysis("zcr"),
    "energy": mf.frame_analysis("energy"),
    "kurtosis": mf.frame_analysis("kurtosis")
}

# Mostrar características
plt.figure(figsize=(12, 10))
for i, (name, (t, values)) in enumerate(features.items(), 1):
    plt.subplot(4, 1, i)
    plt.plot(t, values)
    plt.title(name.capitalize())
    plt.grid(True)
    plt.xlabel("Tiempo [s]")

plt.tight_layout()
plt.show()
