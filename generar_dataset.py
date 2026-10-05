from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
N_CLIENTES = 500

rng = np.random.default_rng(SEMILLA)

NOMBRES = [
    "Laura", "Carlos", "Andrea", "Juan", "Camila", "Santiago", "Valentina", "Mateo",
    "Daniela", "Sebastián", "Natalia", "Felipe", "Paula", "David", "Sofía", "Nicolás",
    "Mariana", "Alejandro", "Juliana", "Andrés", "Carolina", "Diego", "Isabella",
    "Miguel", "Luisa", "Jorge", "Manuela", "Esteban", "Catalina", "Ricardo",
]
APELLIDOS = [
    "Gómez", "Rodríguez", "Martínez", "López", "García", "Hernández", "Pérez",
    "Sánchez", "Ramírez", "Torres", "Díaz", "Vargas", "Castro", "Moreno", "Ortiz",
    "Rojas", "Jiménez", "Mendoza", "Cardona", "Quintero", "Ríos", "Salazar",
    "Acosta", "Parra", "Duarte", "Peña", "Cortés", "Ruiz", "Murillo", "Beltrán",
]


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def generar() -> pd.DataFrame:
    n = N_CLIENTES

    nombres = [
        f"{rng.choice(NOMBRES)} {rng.choice(APELLIDOS)}" for _ in range(n)
    ]
    edad = np.clip(rng.normal(38, 12, n), 18, 75).round().astype(int)

    ingresos = np.exp(rng.normal(np.log(3_200_000), 0.45, n))
    ingresos = (np.clip(ingresos, 1_000_000, 15_000_000) / 50_000).round() * 50_000

    tiempo_cliente = np.clip(rng.exponential(30, n) + 1, 1, 120).round().astype(int)
    satisfaccion = np.clip(rng.normal(6.5, 2.0, n).round(), 1, 10).astype(int)

    frecuencia = np.clip(rng.poisson(1.5 + 0.35 * satisfaccion, n), 0, 15)
    productos = np.clip(rng.poisson(2 + 1.5 * frecuencia, n), 1, 60)

    logit = (
        -0.75
        - 0.62 * (satisfaccion - 6)
        - 0.025 * (tiempo_cliente - 30)
        - 0.28 * (frecuencia - 4)
        - 0.03 * (productos - 8)
        - 0.015 * (edad - 38)
        - 0.12 * (ingresos / 1_000_000 - 3.2)
        + rng.normal(0, 0.6, n)
    )
    abandono = rng.binomial(1, sigmoide(logit))

    df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, n + 1),
            "nombre": nombres,
            "edad": edad,
            "ingresos": ingresos.astype(int),
            "frecuencia_compra": frecuencia,
            "cantidad_productos": productos,
            "tiempo_cliente": tiempo_cliente,
            "satisfaccion": satisfaccion,
            "abandono": abandono,
        }
    )
    return df


def ensuciar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.astype({"edad": float, "ingresos": float, "tiempo_cliente": float, "satisfaccion": float})

    for col, cantidad in [("edad", 3), ("ingresos", 4), ("satisfaccion", 3), ("tiempo_cliente", 2)]:
        idx = rng.choice(df.index, size=cantidad, replace=False)
        df.loc[idx, col] = np.nan

    df.loc[rng.choice(df.index, 2, replace=False), "edad"] = [199, 250]
    df.loc[rng.choice(df.index, 1, replace=False), "satisfaccion"] = 11

    duplicadas = df.sample(5, random_state=SEMILLA)
    df = pd.concat([df, duplicadas], ignore_index=True)
    return df


if __name__ == "__main__":
    carpeta = Path(__file__).parent / "data"
    carpeta.mkdir(exist_ok=True)

    limpio = generar()
    sucio = ensuciar(limpio)
    sucio.to_csv(carpeta / "clientes_raw.csv", index=False)

    print(f"Dataset generado: {len(sucio)} filas (incluye 5 duplicadas) -> data/clientes_raw.csv")
    print(f"Tasa de abandono real: {limpio['abandono'].mean():.1%}")
