import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).parent
RUTA_DATOS = BASE / "data" / "clientes.csv"
RUTA_MODELO = BASE / "ml" / "modelo.pkl"
RUTA_METRICAS = BASE / "ml" / "metricas.json"
RUTA_DB = Path(os.environ.get("CHURN_DB", BASE / "database.db"))

UMBRAL_MEDIO = 0.40
UMBRAL_ALTO = 0.70

COLORES = {"Bajo": "#2F7D5B", "Medio": "#A8741A", "Alto": "#B3362D"}

ETIQUETAS = {
    "edad": "Edad",
    "ingresos": "Ingresos",
    "frecuencia_compra": "Frecuencia de compra",
    "cantidad_productos": "Productos adquiridos",
    "tiempo_cliente": "Tiempo como cliente",
    "satisfaccion": "Satisfacción",
}


def cargar_modelo() -> dict:
    return joblib.load(RUTA_MODELO)


def cargar_metricas() -> dict:
    with open(RUTA_METRICAS, encoding="utf-8") as f:
        return json.load(f)


def cargar_datos() -> pd.DataFrame:
    return pd.read_csv(RUTA_DATOS)


def nivel_riesgo(probabilidad: float) -> str:
    if probabilidad >= UMBRAL_ALTO:
        return "Alto"
    if probabilidad >= UMBRAL_MEDIO:
        return "Medio"
    return "Bajo"


def predecir(modelo: dict, datos: dict) -> float:
    X = pd.DataFrame([datos])[modelo["features"]]
    return float(modelo["pipeline"].predict_proba(X)[0, 1])


def puntuar_dataset(modelo: dict, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["probabilidad"] = modelo["pipeline"].predict_proba(out[modelo["features"]])[:, 1]
    out["riesgo"] = out["probabilidad"].map(nivel_riesgo)
    return out


def efecto_variables(modelo: dict, datos: dict, referencia: pd.DataFrame) -> dict:
    base = predecir(modelo, datos)
    medianas = referencia[modelo["features"]].median()
    efectos = {}
    for col in modelo["features"]:
        alterado = dict(datos)
        alterado[col] = float(medianas[col])
        efectos[col] = base - predecir(modelo, alterado)
    return efectos


def formato_cop(valor) -> str:
    return "$" + f"{int(valor):,}".replace(",", ".")


def _conexion() -> sqlite3.Connection:
    return sqlite3.connect(RUTA_DB)


def init_db() -> None:
    with closing(_conexion()) as con, con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS predicciones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT NOT NULL,
                fecha TEXT NOT NULL,
                edad INTEGER,
                ingresos INTEGER,
                frecuencia_compra INTEGER,
                cantidad_productos INTEGER,
                tiempo_cliente INTEGER,
                satisfaccion INTEGER,
                probabilidad REAL NOT NULL,
                riesgo TEXT NOT NULL,
                resultado TEXT NOT NULL
            )
            """
        )


def guardar_prediccion(nombre: str, datos: dict, probabilidad: float) -> None:
    riesgo = nivel_riesgo(probabilidad)
    with closing(_conexion()) as con, con:
        con.execute(
            """
            INSERT INTO predicciones
            (nombre, fecha, edad, ingresos, frecuencia_compra, cantidad_productos,
             tiempo_cliente, satisfaccion, probabilidad, riesgo, resultado)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nombre,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                int(datos["edad"]),
                int(datos["ingresos"]),
                int(datos["frecuencia_compra"]),
                int(datos["cantidad_productos"]),
                int(datos["tiempo_cliente"]),
                int(datos["satisfaccion"]),
                float(probabilidad),
                riesgo,
                f"Riesgo {riesgo.lower()} de abandono",
            ),
        )


def leer_historial(limite: int | None = None) -> pd.DataFrame:
    consulta = "SELECT * FROM predicciones ORDER BY id DESC"
    if limite:
        consulta += f" LIMIT {int(limite)}"
    with closing(_conexion()) as con:
        return pd.read_sql_query(consulta, con)


def borrar_historial() -> None:
    with closing(_conexion()) as con, con:
        con.execute("DELETE FROM predicciones")
