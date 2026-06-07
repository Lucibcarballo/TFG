import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def graficar_onda(ruta_audio):
    # 1. Cargar el archivo de sonido
    fs, data = wavfile.read(ruta_audio)

    # Si el audio es estéreo, nos quedamos con un solo canal
    if len(data.shape) > 1:
        data = data[:, 0]

    # 2. Normalizar la amplitud entre -1 y 1
    if data.dtype == np.int16:
        data_normalizada = data / 32768.0
    elif data.dtype == np.int32:
        data_normalizada = data / 2147483648.0
    else:
        data_normalizada = data / np.max(np.abs(data))

    # 3. Calcular el vector de tiempo en segundos
    n = len(data_normalizada)
    tiempo = np.arange(n) / fs

    # 4. Extraer el nombre del archivo y aplicar los reemplazos solicitados
    nombre_archivo = os.path.basename(ruta_audio)

    # Creamos un título limpio sustituyendo los nombres de los intérpretes
    titulo_limpio = nombre_archivo.lower()
    titulo_limpio = titulo_limpio.replace("alejandro", "Intérprete 1")
    titulo_limpio = titulo_limpio.replace("uxia", "Intérprete 2")
    # Respetamos el resto del nombre del archivo (como g1_02.wav) y capitalizamos la primera letra
    titulo_limpio = titulo_limpio.capitalize()

    # Creamos el nombre para guardar el archivo .png sin caracteres conflictivos
    nombre_salida_png = nombre_archivo.lower()
    nombre_salida_png = nombre_salida_png.replace("alejandro", "interprete1")
    nombre_salida_png = nombre_salida_png.replace("uxia", "interprete2")
    nombre_sin_extension, _ = os.path.splitext(nombre_salida_png)

    # 5. Configurar y pintar la gráfica al estilo Audacity
    plt.figure(figsize=(12, 4.5))

    # Onda en el azul clásico de Audacity
    plt.plot(tiempo, data_normalizada, color="#4a90e2", linewidth=0.7, alpha=0.9)

    # Límites del gráfico
    plt.xlim(0, tiempo[-1])
    plt.ylim(-1.05, 1.05)

    # Configuración de textos solicitada (Título: 22 con nombre corregido, Ejes: 16)
    plt.title(f"Forma de onda: {titulo_limpio}", fontsize=22, fontweight="bold", pad=15)
    plt.xlabel("Tiempo (segundos)", fontsize=16, labelpad=10)
    plt.ylabel("Amplitud", fontsize=16, labelpad=10)

    # Tamaño de los números de los ejes
    plt.xticks(fontsize=14)
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=14)

    # Líneas guía horizontales
    plt.grid(True, which="both", axis="y", linestyle=":", color="#cccccc", alpha=0.7)
    plt.axhline(0, color="#999999", linewidth=0.8, linestyle="-")

    plt.tight_layout()

    # 6. Guardar la figura en alta definición (.png) con el nombre adaptado
    nombre_final_archivo = f"espectro_{nombre_sin_extension}.png"
    plt.savefig(nombre_final_archivo, dpi=300, bbox_inches="tight")
    print(f"[OK] Gráfica guardada como: {nombre_final_archivo}")

    # Mostrar en pantalla
    plt.show()
    plt.close()


graficar_onda("docs/piezas/guitarra2_uxia pieza.wav")
