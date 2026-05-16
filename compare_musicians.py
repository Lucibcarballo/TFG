import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generar_radar_chart(csv_path):
    # Cargamos datos del CSV generado por procesa_guitarras.py
    df = pd.read_csv(csv_path)

    cols_excluir = ["Archivo", "TNR", "PR"]
    # Seleccionar solo columnas numéricas
    cols_metrics = [c for c in df.columns if c not in cols_excluir and c != "Clase"]

    # 1. ESCALAR PRIMERO LOS DATOS RAW (Corrección clave)
    # Esto asegura que el 0 y el 1 se calculan sobre todas las notas de ambos intérpretes
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[cols_metrics] = scaler.fit_transform(df[cols_metrics])

    # 2. AGRUPAR Y CALCULAR LA MEDIA (después de escalar)
    df_mean = df_scaled.groupby("Clase")[cols_metrics].mean().reset_index()

    # Configuración del gráfico
    labels = cols_metrics
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in df_mean.iterrows():
        values = row[cols_metrics].values.flatten().tolist()
        values += values[:1]  # Cerrar el círculo
        ax.plot(angles, values, linewidth=2, label=row["Clase"])
        ax.fill(angles, values, alpha=0.25)

    # Etiquetas de los ejes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    plt.title("Comparativa de Intérpretes: Uxía vs Alejandro", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    nombre_archivo = "radar_chart_comparativa_interpretes.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches="tight")
    print(f"Gráfico guardado con éxito como: {nombre_archivo}")

    plt.show()


def main():
    generar_radar_chart("dataset_guitarras_grabaciones_notas.csv")


if __name__ == "__main__":
    main()
