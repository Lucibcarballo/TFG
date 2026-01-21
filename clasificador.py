import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# __________________________configuración__________________________
ruta_csv = r"C:\Users\lucib\Desktop\TFG\audio\MIS_AUDIOS\dataset_final.csv"


def cargar_dataset(ruta_csv):

    df = pd.read_csv(ruta_csv)
    if "filename" not in df.columns or "clase" not in df.columns:
        raise ValueError(
            "El CSV debe contener al menos las columnas 'filename' y 'clase'."
        )
    print(f"Dataset cargado. Total de notas: {len(df)}")

    return df


def train_model(df):
    X = df.drop(["filename", "clase"], axis=1)  # X son las características
    y = df["clase"]  # y es la etiqueta (electrica/española)
    print(X.shape, y.shape)

    # etiquetas 'electrica'/'española' en numeros (0 y 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y_encoded, test_size=0.2, random_state=42
    # )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)
    model.fit(X, y_encoded)

    # y_pred = model.predict(X_test)
    # print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred, target_names=le.classes_))

    y_pred = model.predict(X)
    print(f"\nAccuracy: {accuracy_score(y, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))

    feature_importance = pd.DataFrame(
        {"caracteristica": X.columns, "importancia": model.feature_importances_}
    ).sort_values(by="importancia", ascending=False)

    print("\nImportancia de los parámetros en la clasificación:\n")
    print(feature_importance)

    # return y_test, y_pred, le, feature_importance
    return y, y_pred, le, feature_importance


def graficas(y_test, y_pred, le, feature_importance):
    # Matriz de Confusión
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Clase Real")

    # Importancia de Características (Bar Chart)
    plt.figure(figsize=(6, 6))
    sns.barplot(
        x="importancia", y="caracteristica", data=feature_importance, palette="viridis"
    )
    plt.title("Importancia de los Parámetros Acústicos")
    plt.xlabel("Importancia Relativa")
    plt.ylabel("Parámetro")

    plt.tight_layout()
    plt.show()


def main():
    df = cargar_dataset(ruta_csv)
    y_test, y_pred, le, feature_importance = train_model(df)
    graficas(y_test, y_pred, le, feature_importance)


if __name__ == "__main__":
    main()
