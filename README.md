# ChurnGuard: predicción de abandono de clientes

Aplicación web de Machine Learning, desarrollada solo con Python, que estima la probabilidad de que un cliente de una empresa de servicios abandone y la presenta en un dashboard.

Flujo: formulario web, modelo de ML, predicción, resultado visual y registro en historial (SQLite).

## Ejecución

```bash
pip install -r requirements.txt
python generar_dataset.py
python entrenar.py
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`. Si se reentrena el modelo, hay que reiniciar la aplicación.

## Estructura

| Archivo | Contenido |
|---|---|
| `generar_dataset.py` | Crea `data/clientes_raw.csv` con 505 filas, nulos y duplicados |
| `entrenar.py` | Exploración con Pandas, limpieza, tres modelos, métricas y guardado del mejor |
| `app.py` | Interfaz web en Streamlit: dashboard, predicción, historial y análisis del modelo |
| `utils.py` | Carga del modelo, nivel de riesgo y base de datos SQLite |
| `data/` | Datos originales (`clientes_raw.csv`) y datos limpios (`clientes.csv`) |
| `ml/` | Modelo entrenado (`modelo.pkl`) y resultados (`metricas.json`) |
| `assets/` | Logo e ícono |
| `.streamlit/config.toml` | Tema visual |

## Cumplimiento del enunciado

| Requisito | Ubicación |
|---|---|
| Dataset de al menos 150 registros | `data/clientes.csv`, 500 registros |
| Análisis con Pandas | `entrenar.py`, sección 1 |
| Limpieza y preparación | `entrenar.py`, sección 2 |
| Variables X e y | `entrenar.py`, sección 3 |
| División entrenamiento y prueba | `entrenar.py`, sección 4 (80 % y 20 %, estratificada) |
| Al menos dos modelos de clasificación | Regresión Logística, Random Forest y KNN |
| Accuracy, matriz de confusión y reporte | `entrenar.py`, sección 6, y página Análisis del modelo |
| Selección del modelo | `entrenar.py`, sección 7, guardado en `ml/modelo.pkl` |
| Dashboard, tarjetas, menú y logo | Página Dashboard |
| Formulario, botón Predecir y resultado visual | Página Nueva predicción |
| Al menos dos gráficas | Dashboard con cuatro gráficas y dos más en Nueva predicción |
| Historial con nombre, fecha, datos, resultado y probabilidad | Página Historial |
| Diseño responsive | Cuadrícula adaptable y reglas CSS para pantallas pequeñas |

## Conceptos clave

- El modelo se entrena una sola vez con `entrenar.py` y se guarda con `joblib`. La aplicación solo lo carga y predice.
- El `StandardScaler` va dentro del `Pipeline` guardado, de modo que la aplicación aplica el mismo escalado del entrenamiento.
- Se usa `predict_proba` para obtener una probabilidad. El nivel de riesgo es Bajo por debajo de 40 %, Medio entre 40 % y 70 %, y Alto desde 70 %.
- Las columnas se reordenan con `modelo["features"]` antes de predecir, porque un orden distinto produce resultados erróneos sin mostrar errores.
- Los datos son sintéticos. El abandono depende de la satisfacción, la antigüedad y la frecuencia de compra, más ruido aleatorio. Por eso el accuracy ronda el 75 % y no el 100 %.
- Con 100 casos de prueba, el accuracy varía unos puntos entre ejecuciones. La validación cruzada de `metricas.json` es una estimación más estable.
- En la página de predicción, el efecto de cada variable se calcula reemplazando su valor por la mediana de la base y midiendo cuánto cambia la probabilidad.

## Uso de un dataset real

Reemplazar `data/clientes_raw.csv` por un archivo con las columnas `edad`, `ingresos`, `frecuencia_compra`, `cantidad_productos`, `tiempo_cliente`, `satisfaccion` y `abandono` (1 si el cliente se fue, 0 si permanece), y ejecutar de nuevo `entrenar.py`.
