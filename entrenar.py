"""
entrenar.py
-----------
Pipeline completo de Machine Learning para predecir el abandono de clientes:

  1. Carga y exploración con Pandas
  2. Limpieza y preparación
  3. Definición de X (variables) y y (objetivo)
  4. División entrenamiento / prueba
  5. Entrenamiento de 3 modelos de clasificación
  6. Evaluación: accuracy, matriz de confusión, reporte de clasificación
  7. Selección del mejor modelo y guardado (ml/modelo.pkl)

Uso:  python entrenar.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
SEMILLA = 42

FEATURES = [
    "edad",
    "ingresos",
    "frecuencia_compra",
    "cantidad_productos",
    "tiempo_cliente",
    "satisfaccion",
]
OBJETIVO = "abandono"


def titulo(texto: str) -> None:
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. CARGA Y EXPLORACIÓN
# ---------------------------------------------------------------------------
titulo("1. EXPLORACIÓN DEL DATASET (Pandas)")
df = pd.read_csv(BASE / "data" / "clientes_raw.csv")
print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")
print("\nPrimeras filas:")
print(df.head())
print("\nTipos de dato:")
print(df.dtypes)
print("\nEstadísticas descriptivas:")
print(df[FEATURES].describe().round(2))
print("\nValores nulos por columna:")
print(df.isna().sum())
print(f"\nFilas duplicadas: {df.duplicated().sum()}")

# ---------------------------------------------------------------------------
# 2. LIMPIEZA Y PREPARACIÓN
# ---------------------------------------------------------------------------
titulo("2. LIMPIEZA Y PREPARACIÓN")
filas_inicio = len(df)

# a) Duplicados
df = df.drop_duplicates()
print(f"- Duplicados eliminados: {filas_inicio - len(df)}")

# b) Valores imposibles -> se convierten en nulos para imputarlos después
imposibles_edad = ((df["edad"] < 18) | (df["edad"] > 100)).sum()
imposibles_sat = ((df["satisfaccion"] < 1) | (df["satisfaccion"] > 10)).sum()
df.loc[(df["edad"] < 18) | (df["edad"] > 100), "edad"] = np.nan
df.loc[(df["satisfaccion"] < 1) | (df["satisfaccion"] > 10), "satisfaccion"] = np.nan
print(f"- Edades imposibles (<18 o >100) marcadas como nulas: {imposibles_edad}")
print(f"- Satisfacción fuera de rango (1-10) marcada como nula: {imposibles_sat}")

# c) Nulos -> se rellenan con la MEDIANA (no se afecta por valores extremos)
nulos = df[FEATURES].isna().sum().sum()
for col in FEATURES:
    df[col] = df[col].fillna(df[col].median())
print(f"- Valores nulos imputados con la mediana: {nulos}")

# d) Tipos de dato correctos
for col in ["edad", "ingresos", "tiempo_cliente", "satisfaccion"]:
    df[col] = df[col].round().astype(int)

assert df[FEATURES].isna().sum().sum() == 0, "Quedaron nulos sin tratar"
df.to_csv(BASE / "data" / "clientes.csv", index=False)
print(f"\nDataset limpio: {len(df)} filas -> data/clientes.csv")
print(f"Distribución del objetivo:\n{df[OBJETIVO].value_counts().rename({0: 'Permanece', 1: 'Abandona'})}")

# ---------------------------------------------------------------------------
# 3. VARIABLES X e y  +  4. DIVISIÓN ENTRENAMIENTO / PRUEBA
# ---------------------------------------------------------------------------
titulo("3 y 4. VARIABLES X, y Y DIVISIÓN TRAIN/TEST")
X = df[FEATURES]  # lo que el modelo usa para decidir
y = df[OBJETIVO]  # lo que queremos predecir (1 = abandona, 0 = permanece)

# stratify=y mantiene la misma proporción de abandonos en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEMILLA, stratify=y
)
print(f"X: {X.shape} | y: {y.shape}")
print(f"Entrenamiento: {len(X_train)} filas (80%) | Prueba: {len(X_test)} filas (20%)")

# ---------------------------------------------------------------------------
# 5. ENTRENAMIENTO DE MODELOS
# ---------------------------------------------------------------------------
titulo("5. ENTRENAMIENTO DE MODELOS")
# Los modelos basados en distancia o coeficientes (KNN, Regresión Logística) necesitan
# variables en la misma escala. Por eso el StandardScaler va DENTRO del Pipeline:
# así viaja junto al modelo y la app web aplica exactamente la misma transformación.
modelos = {
    "Regresión Logística": Pipeline(
        [("escalador", StandardScaler()), ("modelo", LogisticRegression(max_iter=1000, random_state=SEMILLA))]
    ),
    "Random Forest": Pipeline(
        [("modelo", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=SEMILLA))]
    ),
    "K-Nearest Neighbors": Pipeline(
        [("escalador", StandardScaler()), ("modelo", KNeighborsClassifier(n_neighbors=9))]
    ),
}

# ---------------------------------------------------------------------------
# 6. EVALUACIÓN
# ---------------------------------------------------------------------------
resultados = {}
for nombre, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    resultados[nombre] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_accuracy_media": cv.mean(),
        "cv_accuracy_std": cv.std(),
        "matriz_confusion": confusion_matrix(y_test, pred).tolist(),
        "reporte": classification_report(
            y_test, pred, target_names=["Permanece", "Abandona"], output_dict=True
        ),
    }

    titulo(f"MODELO: {nombre}")
    print(f"Accuracy (prueba):          {resultados[nombre]['accuracy']:.4f}")
    print(f"Accuracy validación cruzada: {cv.mean():.4f} (+/- {cv.std():.4f})")
    print(f"ROC-AUC:                     {resultados[nombre]['roc_auc']:.4f}")
    print("\nMatriz de confusión  [filas = real, columnas = predicho]")
    print(pd.DataFrame(
        resultados[nombre]["matriz_confusion"],
        index=["Real: Permanece", "Real: Abandona"],
        columns=["Pred: Permanece", "Pred: Abandona"],
    ))
    print("\nReporte de clasificación:")
    print(classification_report(y_test, pred, target_names=["Permanece", "Abandona"]))

# ---------------------------------------------------------------------------
# 7. COMPARACIÓN Y SELECCIÓN DEL MODELO FINAL
# ---------------------------------------------------------------------------
titulo("7. COMPARACIÓN Y SELECCIÓN")
tabla = pd.DataFrame(resultados).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_accuracy_media"]]
tabla = tabla.astype(float).round(4)
print(tabla.sort_values("accuracy", ascending=False))

# Criterio: mayor accuracy en prueba; si empatan, gana el de mayor F1 en la clase "Abandona"
# (porque detectar a quienes se van es el objetivo del negocio).
mejor = max(resultados, key=lambda k: (round(resultados[k]["accuracy"], 4), resultados[k]["f1"]))
print(f"\n>>> MODELO SELECCIONADO: {mejor}  (accuracy = {resultados[mejor]['accuracy']:.2%})")

# Importancia de variables del modelo elegido (funciona para cualquier tipo de modelo)
perm = permutation_importance(
    modelos[mejor], X_test, y_test, n_repeats=30, random_state=SEMILLA, scoring="accuracy"
)
importancias = dict(sorted(zip(FEATURES, perm.importances_mean.round(4).tolist()), key=lambda t: -t[1]))
print("\nImportancia de variables (permutación):")
for variable, valor in importancias.items():
    print(f"  {variable:<20} {valor:.4f}")

# ---------------------------------------------------------------------------
# GUARDADO: el modelo + el orden de las columnas (clave para no equivocarse al predecir)
# ---------------------------------------------------------------------------
(BASE / "ml").mkdir(exist_ok=True)
joblib.dump({"pipeline": modelos[mejor], "features": FEATURES, "nombre": mejor}, BASE / "ml" / "modelo.pkl")

metricas = {
    "modelo_seleccionado": mejor,
    "criterio": "Mayor accuracy en el conjunto de prueba (desempate por F1 de la clase Abandona)",
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "features": FEATURES,
    "tasa_abandono": float(y.mean()),
    "resultados": resultados,
    "importancias": importancias,
}
with open(BASE / "ml" / "metricas.json", "w", encoding="utf-8") as f:
    json.dump(metricas, f, ensure_ascii=False, indent=2)

print("\nArchivos guardados: ml/modelo.pkl y ml/metricas.json")
