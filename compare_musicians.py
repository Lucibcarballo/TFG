import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def generar_radar_chart(
    csv_path,
    clase_columna="Clase",
    es_encuesta=False,
    titulo="Comparativa de Intérpretes",
    archivo_salida="radar_chart.png",
):
    # Cargamos datos del CSV
    df = pd.read_csv(csv_path)

    # Seleccionar solo columnas numéricas (excluyendo la clase y variables de control del software)
    cols_excluir = ["Archivo", "TNR", "PR", clase_columna]
    cols_metrics = [c for c in df.columns if c not in cols_excluir]

    if not es_encuesta:
        # para resultados codigo: ESCALAR PRIMERO LOS DATOS RAW
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[cols_metrics] = scaler.fit_transform(df[cols_metrics])
        # Agrupar y calcular media
        df_mean = df_scaled.groupby(clase_columna)[cols_metrics].mean().reset_index()
    else:
        # para resultados encuesta: NO ESCALAR (Ya vienen del 0 al 10)
        df_mean = df.groupby(clase_columna)[cols_metrics].mean().reset_index()

    # Configuración del gráfico
    num_vars = len(cols_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Cerrar el círculo

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, row in df_mean.iterrows():
        values = row[cols_metrics].values.flatten().tolist()
        values += values[:1]  # Cerrar el círculo
        ax.plot(angles, values, linewidth=2, label=row[clase_columna])
        ax.fill(angles, values, alpha=0.25)

    # Etiquetas de los ejes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), cols_metrics)

    # LÍMITES DE ESCALA DEPENDIENDO DEL DATO
    if es_encuesta:
        ax.set_ylim(0, 10)
        ax.set_rticks([2, 4, 6, 8, 10])
    else:
        ax.set_ylim(0, 1)  # El MinMaxScaler deja todo entre 0 y 1

    plt.title(titulo, size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    plt.savefig(archivo_salida, dpi=300, bbox_inches="tight")
    print(f"[OK] Gráfico guardado con éxito como: {archivo_salida}")

    plt.show()


def main():
    generar_radar_chart(
        csv_path="c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\notas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_notas.csv",
        clase_columna="Clase",
        es_encuesta=False,
        titulo="Comparativa Software: Intérprete 1 vs Intérprete 2",  # 1 uxia, 2 alejandro
        archivo_salida="radar_chart_objetivo_notas.png",
    )

    generar_radar_chart(
        csv_path="c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\piezas_grabaciones_reducc_ruido_12_marzo\\resultados_completos\\dataset_guitarras_grabaciones_global.csv",
        clase_columna="Clase",
        es_encuesta=False,
        titulo="Comparativa Software: Intérprete 1 vs Intérprete 2",  # 1 uxia, 2 alejandro
        archivo_salida="radar_chart_objetivo_piezas.png",
    )

    generar_radar_chart(
        csv_path="c:\\Users\\lucib\\Desktop\\TFG\\RESULTADOS\\encuestas\\datos_encuesta_piezas.csv",
        clase_columna="Intérprete",
        es_encuesta=True,
        titulo="Comparativa Encuesta: Uxía vs Alejandro",
        archivo_salida="radar_chart_subjetivo_piezas.png",
    )


if __name__ == "__main__":
    main()
