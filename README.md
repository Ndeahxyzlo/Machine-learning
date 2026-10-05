# 🛡️ ChurnGuard — Predicción de abandono de clientes

Aplicación web de Machine Learning, hecha 100 % en Python, que predice qué clientes de una
empresa de servicios tienen riesgo de abandonar, y lo muestra en un dashboard profesional.

**Flujo:** Formulario web → Modelo de ML → Predicción → Resultado visual → Registro en historial (SQLite)

---

## Cómo ejecutarlo

```bash
# 1. Instalar dependencias (una sola vez)
pip install -r requirements.txt

# 2. Generar el dataset (500 clientes, "sucio" a propósito)
python generar_dataset.py

# 3. Limpiar, entrenar y comparar los modelos (genera ml/modelo.pkl)
python entrenar.py

# 4. Abrir la aplicación web
streamlit run app.py
```

Se abre en el navegador en `http://localhost:8501`.
Si cambias el dataset o reentrenas, **reinicia la app** (Ctrl+C y volver a ejecutar el paso 4).

---

## Estructura del proyecto

```
churn-app/
├── generar_dataset.py   # crea data/clientes_raw.csv (505 filas con nulos y duplicados)
├── entrenar.py          # Pandas + limpieza + 3 modelos + métricas + guarda el mejor
├── app.py               # interfaz web (Streamlit): dashboard, predicción, historial, modelo
├── utils.py             # lógica: cargar modelo, nivel de riesgo, base de datos SQLite
├── requirements.txt
├── data/
│   ├── clientes_raw.csv # datos originales (sucios)
│   └── clientes.csv     # datos limpios (500 filas)
├── ml/
│   ├── modelo.pkl       # modelo entrenado que usa la app
│   └── metricas.json    # resultados de la comparación de modelos
├── assets/              # logo e ícono
├── .streamlit/config.toml  # colores y tema
└── database.db          # historial de predicciones (se crea sola)
```

---

## Cumplimiento del enunciado

| Requisito | Dónde está |
|---|---|
| Dataset con mínimo 150 registros | `data/clientes.csv` → 500 registros |
| Análisis con Pandas | `entrenar.py` sección 1 (`head`, `dtypes`, `describe`, nulos, duplicados) |
| Limpieza y preparación | `entrenar.py` sección 2 (duplicados, valores imposibles, nulos → mediana) |
| Variables X e y | `entrenar.py` sección 3 |
| División entrenamiento / prueba | `entrenar.py` sección 4 (80 % / 20 %, estratificada) |
| Mínimo 2 modelos de clasificación | Regresión Logística, Random Forest y KNN |
| Accuracy, matriz de confusión, reporte | `entrenar.py` sección 6 y página **Análisis del modelo** de la app |
| Comparación y selección del modelo | `entrenar.py` sección 7 → se guarda en `ml/modelo.pkl` |
| Dashboard, tarjetas, menú, logo | Página **Dashboard** |
| Formulario + botón Predecir + resultado visual | Página **Nueva predicción** |
| Gráficas (mínimo 2) | Dashboard: 4 gráficas · Predicción: medidor y radar |
| Historial (nombre, fecha, datos, resultado, probabilidad) | Página **Historial** (SQLite) |
| Diseño responsive | Cuadrícula adaptable + CSS para celular |

---

## Conceptos clave (para la sustentación)

- **El modelo se entrena una sola vez** (`entrenar.py`) y se guarda con `joblib`. La app solo lo
  *carga* y predice; nunca entrena.
- **Pipeline:** el `StandardScaler` va dentro del modelo guardado. Así la app aplica exactamente
  el mismo escalado que en el entrenamiento.
- **Probabilidad, no solo 0/1:** se usa `predict_proba` para mostrar "82 %". Con eso se define el
  nivel de riesgo: **Bajo < 40 %**, **Medio 40–70 %**, **Alto ≥ 70 %**.
- **Orden de columnas:** se reordenan con `modelo["features"]` antes de predecir. Si el orden cambia,
  el modelo da resultados absurdos sin mostrar ningún error.
- **Datos sintéticos con lógica:** el abandono depende de satisfacción, antigüedad, frecuencia, etc.
  más ruido aleatorio. Por eso el accuracy ronda el **75 %** y no 100 %: un resultado perfecto en
  datos reales sería señal de *overfitting* o de fuga de información.
- **Limitación honesta:** con solo 100 casos de prueba, el accuracy varía unos puntos entre
  ejecuciones. La validación cruzada (en `metricas.json`) da una estimación más estable.

---

## Usar un dataset real

Reemplaza `data/clientes_raw.csv` por tu archivo manteniendo estas columnas:
`edad, ingresos, frecuencia_compra, cantidad_productos, tiempo_cliente, satisfaccion, abandono`
(`abandono` = 1 si el cliente se fue, 0 si permanece) y vuelve a ejecutar `entrenar.py`.
