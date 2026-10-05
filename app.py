"""
app.py
------
ChurnGuard · Aplicación web de Machine Learning para predecir el abandono de clientes.

Flujo:  Formulario -> Modelo de ML -> Predicción -> Resultado visual -> Historial (SQLite)

Ejecutar:  streamlit run app.py
"""
import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

BASE = Path(__file__).parent

st.set_page_config(
    page_title="ChurnGuard · Predicción de abandono",
    page_icon=str(BASE / "assets" / "icono.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# ESTILOS (CSS)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp, .stApp button, .stApp input, .stApp textarea { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
footer { visibility: hidden; }

/* ---- Encabezado de página ---- */
.page-head h1 { font-size: 1.9rem; font-weight: 800; color: #111827; margin: 0; padding: 0; letter-spacing: -.02em; }
.page-head p  { color: #6B7280; margin: .25rem 0 1.4rem; font-size: .98rem; }

/* ---- Tarjetas KPI (grid responsive) ---- */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 16px; margin: 0 0 20px; }
.kpi { background: #fff; border: 1px solid #E8EBF3; border-radius: 16px; padding: 18px 20px;
       display: flex; gap: 14px; align-items: center;
       box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 16px rgba(16,24,40,.04); }
.kpi-icon { width: 50px; height: 50px; border-radius: 14px; display: flex; align-items: center; justify-content: center; flex: none; }
.kpi-icon svg { width: 25px; height: 25px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.kpi-label { font-size: .82rem; color: #6B7280; font-weight: 500; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #111827; line-height: 1.15; letter-spacing: -.02em; }
.kpi-sub   { font-size: .76rem; color: #9CA3AF; margin-top: 2px; }

/* ---- Paneles (contenedores con key="panel_...") ---- */
[class*="st-key-panel"] { background: #fff; border: 1px solid #E8EBF3; border-radius: 16px; padding: 1.1rem 1.3rem 1rem;
       box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 6px 16px rgba(16,24,40,.04); }
.panel-title { font-weight: 700; font-size: 1.02rem; color: #111827; margin: 0; }
.panel-sub   { font-size: .8rem; color: #9CA3AF; margin: 0 0 .35rem; }

/* ---- Formulario ---- */
.stApp label p { font-weight: 600; font-size: .86rem; color: #374151; }
div[data-baseweb="input"], div[data-baseweb="base-input"] { border-radius: 10px; }
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5); border: 0; border-radius: 12px;
    font-weight: 700; padding: .65rem 1rem; box-shadow: 0 8px 18px rgba(79,70,229,.30); }
[data-testid="stBaseButton-secondary"] { border-radius: 10px; font-weight: 600; }

/* ---- Resultado de la predicción ---- */
.result-card { border-radius: 16px; padding: 22px 24px; border: 1px solid var(--c); background: var(--bg); }
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.badge { display: inline-block; padding: .3rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 700; background: var(--c); color: #fff; }
.result-name { font-weight: 700; color: #111827; font-size: 1rem; }
.result-title { font-size: 1.55rem; font-weight: 800; color: #111827; margin: 14px 0 2px; letter-spacing: -.02em; }
.result-prob { color: #4B5563; font-size: .95rem; }
.result-prob span { font-size: 2.9rem; font-weight: 800; color: var(--c); letter-spacing: -.03em; margin-right: 6px; }
.bar { height: 14px; background: rgba(255,255,255,.85); border-radius: 999px; overflow: hidden; margin: 12px 0 6px; border: 1px solid rgba(0,0,0,.05); }
.bar-fill { height: 100%; border-radius: 999px; background: var(--c); }
.bar-scale { display: flex; font-size: .72rem; color: #6B7280; font-weight: 600; }
.bar-scale span { text-align: center; }
.result-reco { margin: 14px 0 0; padding-top: 12px; border-top: 1px dashed rgba(0,0,0,.12); color: #374151; font-size: .92rem; }

.chip { display: inline-block; padding: .3rem .75rem; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 0 .35rem .4rem 0; }
.chip-mal { background: #FEF2F2; color: #B91C1C; }
.chip-bien { background: #ECFDF5; color: #047857; }

.vacio { text-align: center; padding: 3rem 1rem; color: #6B7280; }
.vacio .big { font-size: 2.6rem; }
.vacio b { color: #111827; display: block; font-size: 1.05rem; margin: .4rem 0 .2rem; }

/* ---- Menú lateral ---- */
[data-testid="stSidebarNav"] a { border-radius: 10px; }
.side-card { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
             padding: 12px 14px; font-size: .78rem; line-height: 1.5; color: #C7D2FE; }
.side-card b { color: #fff; }

/* ---- Celular ---- */
@media (max-width: 640px) {
    .block-container { padding: 1.2rem .8rem 3rem; }
    .page-head h1 { font-size: 1.5rem; }
    .kpi-value { font-size: 1.5rem; }
    .result-prob span { font-size: 2.3rem; }
    .result-title { font-size: 1.3rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.logo(str(BASE / "assets" / "logo.svg"), icon_image=str(BASE / "assets" / "icono.svg"), size="large")

# Iconos SVG (estilo "lucide") para las tarjetas
ICON = {
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "alert": '<path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
}

AZUL, VERDE, AMBAR, ROJO = "#4F46E5", "#10B981", "#F59E0B", "#EF4444"
ESTADO_COLORES = {"Permanece": "#6366F1", "Abandona": "#F43F5E"}
FONDO = {"Bajo": "#ECFDF5", "Medio": "#FFFBEB", "Alto": "#FEF2F2"}

RECOMENDACION = {
    "Alto": "Contactar al cliente en las próximas 48 horas con una oferta de retención personalizada y revisar su experiencia reciente.",
    "Medio": "Hacer seguimiento: enviar una encuesta de satisfacción y ofrecer beneficios de fidelización antes de que el riesgo suba.",
    "Bajo": "Cliente estable. Mantener la comunicación regular y considerar programas de recompensa para reforzar la relación.",
}


# ---------------------------------------------------------------------------
# CARGA (con caché para que la app sea rápida)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_modelo():
    return utils.cargar_modelo()


@st.cache_data
def get_metricas():
    return utils.cargar_metricas()


@st.cache_data
def get_datos():
    """Dataset completo con la probabilidad y el nivel de riesgo que calcula el modelo."""
    df = utils.puntuar_dataset(get_modelo(), utils.cargar_datos())
    df["estado"] = df["abandono"].map({0: "Permanece", 1: "Abandona"})
    return df


utils.init_db()


# ---------------------------------------------------------------------------
# COMPONENTES REUTILIZABLES
# ---------------------------------------------------------------------------
def cabecera(titulo: str, subtitulo: str) -> None:
    st.markdown(f'<div class="page-head"><h1>{titulo}</h1><p>{subtitulo}</p></div>', unsafe_allow_html=True)


def tarjetas(items: list[tuple]) -> None:
    """items: (icono, color de fondo, color del icono, etiqueta, valor, subtítulo)"""
    cuerpo = ""
    for icono, fondo, color, etiqueta, valor, sub in items:
        cuerpo += (
            f'<div class="kpi"><div class="kpi-icon" style="background:{fondo};color:{color}">'
            f'<svg viewBox="0 0 24 24">{ICON[icono]}</svg></div>'
            f'<div><div class="kpi-label">{etiqueta}</div><div class="kpi-value">{valor}</div>'
            f'<div class="kpi-sub">{sub}</div></div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{cuerpo}</div>', unsafe_allow_html=True)


def titulo_panel(titulo: str, sub: str = "") -> None:
    st.markdown(f'<p class="panel-title">{titulo}</p><p class="panel-sub">{sub}</p>', unsafe_allow_html=True)


def estilo(fig: go.Figure, alto: int = 310) -> go.Figure:
    fig.update_layout(
        height=alto,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12.5, color="#374151"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0.5, xanchor="center", title=None),
        hoverlabel=dict(font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E5E7EB")
    fig.update_yaxes(gridcolor="#EEF0F6", zeroline=False)
    return fig


def mostrar(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def vacio(emoji: str, titulo: str, texto: str) -> None:
    st.markdown(f'<div class="vacio"><div class="big">{emoji}</div><b>{titulo}</b>{texto}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PÁGINA 1: DASHBOARD
# ---------------------------------------------------------------------------
def pagina_dashboard():
    datos = get_datos()
    historial = utils.leer_historial()

    cabecera("Dashboard", "Resumen general del riesgo de abandono en la base de clientes.")

    total = len(datos)
    n_alto = int((datos["riesgo"] == "Alto").sum())
    n_bajo = int((datos["riesgo"] == "Bajo").sum())
    promedio = datos["probabilidad"].mean()

    tarjetas([
        ("users", "#EEF2FF", AZUL, "Total de clientes analizados", f"{total:,}".replace(",", "."),
         f"{len(historial)} predicciones nuevas en el historial"),
        ("alert", "#FEF2F2", ROJO, "Clientes con riesgo alto", f"{n_alto}", f"{n_alto / total:.1%} de la base"),
        ("check", "#ECFDF5", VERDE, "Clientes con riesgo bajo", f"{n_bajo}", f"{n_bajo / total:.1%} de la base"),
        ("activity", "#FFFBEB", AMBAR, "Prob. promedio de abandono", f"{promedio:.1%}", "Promedio estimado por el modelo"),
    ])

    # ---- Fila 1 de gráficas ----
    c1, c2 = st.columns(2)
    with c1, st.container(key="panel_riesgo"):
        titulo_panel("Clientes por nivel de riesgo", "Bajo < 40% · Medio 40–70% · Alto ≥ 70%")
        conteo = datos["riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"]).fillna(0).reset_index()
        conteo.columns = ["riesgo", "clientes"]
        fig = px.pie(conteo, names="riesgo", values="clientes", hole=0.64, color="riesgo", color_discrete_map=utils.COLORES)
        fig.update_traces(textinfo="percent", textfont_size=13, sort=False, marker=dict(line=dict(color="#fff", width=3)))
        fig.add_annotation(text=f"<b style='font-size:26px'>{total}</b><br>clientes", showarrow=False)
        mostrar(estilo(fig))

    with c2, st.container(key="panel_satisfaccion"):
        titulo_panel("Nivel de satisfacción", "Comparación entre clientes que abandonaron y que permanecieron")
        fig = px.histogram(datos, x="satisfaccion", color="estado", barmode="group", nbins=10,
                           color_discrete_map=ESTADO_COLORES, labels={"satisfaccion": "Satisfacción (1–10)", "count": "Clientes"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.25, yaxis_title="Clientes")
        mostrar(estilo(fig))

    st.write("")
    # ---- Fila 2 de gráficas ----
    c3, c4 = st.columns(2)
    with c3, st.container(key="panel_edades"):
        titulo_panel("Distribución de edades", "Cantidad de clientes por rango de edad")
        fig = px.histogram(datos, x="edad", nbins=14, color_discrete_sequence=["#6366F1"], labels={"edad": "Edad (años)"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.08, yaxis_title="Clientes")
        mostrar(estilo(fig))

    with c4, st.container(key="panel_antiguedad"):
        titulo_panel("Abandono según antigüedad", "% de clientes que abandonan por tiempo en la empresa")
        tramos = pd.cut(datos["tiempo_cliente"], bins=[0, 6, 12, 24, 48, 1000],
                        labels=["0–6 meses", "6–12 meses", "1–2 años", "2–4 años", "+4 años"])
        tasa = (datos.groupby(tramos, observed=True)["abandono"].mean() * 100).round(1).reset_index()
        tasa.columns = ["tramo", "tasa"]
        fig = px.bar(tasa, x="tramo", y="tasa", text="tasa", color="tasa",
                     color_continuous_scale=["#A7F3D0", "#FDE68A", "#FCA5A5"], labels={"tramo": "", "tasa": "% abandono"})
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", marker_line_width=0, cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, bargap=0.3, yaxis_range=[0, max(tasa["tasa"]) * 1.25])
        mostrar(estilo(fig))

    st.write("")
    # ---- Últimas predicciones ----
    with st.container(key="panel_ultimas"):
        titulo_panel("Últimas predicciones realizadas", "Las 5 más recientes del historial")
        if historial.empty:
            vacio("🔮", "Aún no hay predicciones", "Ve a <b style='display:inline;font-size:inherit'>Nueva predicción</b> para analizar tu primer cliente.")
        else:
            tabla = historial.head(5).copy()
            tabla["Riesgo"] = tabla["riesgo"].map(lambda r: f"{utils.ICONOS[r]} {r}")
            tabla["Probabilidad"] = tabla["probabilidad"] * 100
            st.dataframe(
                tabla[["nombre", "fecha", "satisfaccion", "Riesgo", "Probabilidad"]].rename(
                    columns={"nombre": "Cliente", "fecha": "Fecha", "satisfaccion": "Satisfacción"}),
                hide_index=True, width="stretch",
                column_config={"Probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad de abandono", format="%.0f%%", min_value=0, max_value=100)},
            )


# ---------------------------------------------------------------------------
# PÁGINA 2: NUEVA PREDICCIÓN
# ---------------------------------------------------------------------------
EJEMPLOS = {
    "alto": dict(f_nombre="Laura Gómez", f_edad=27, f_ingresos=2_000_000, f_frecuencia=2,
                 f_productos=5, f_tiempo=10, f_satisfaccion=5),
    "medio": dict(f_nombre="Mariana Torres", f_edad=35, f_ingresos=3_000_000, f_frecuencia=3,
                  f_productos=8, f_tiempo=14, f_satisfaccion=6),
    "bajo": dict(f_nombre="Carlos Rodríguez", f_edad=45, f_ingresos=4_200_000, f_frecuencia=4,
                 f_productos=10, f_tiempo=36, f_satisfaccion=7),
}


def cargar_ejemplo(clave: str) -> None:
    for k, v in EJEMPLOS[clave].items():
        st.session_state[k] = v


def senales(datos: dict, df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Compara al cliente contra el promedio de los que permanecen, usando el mismo dataset."""
    perm, aban = df[df["abandono"] == 0], df[df["abandono"] == 1]
    malas, buenas = [], []
    for col in features:
        desv = df[col].std()
        sentido = 1 if aban[col].mean() > perm[col].mean() else -1  # hacia dónde se mueven los que se van
        if abs(aban[col].mean() - perm[col].mean()) / desv < 0.15:
            continue  # variable que casi no diferencia a unos de otros
        diferencia = (datos[col] - perm[col].mean()) / desv * sentido
        valor = utils.formato_cop(datos[col]) if col == "ingresos" else f"{datos[col]:g}"
        ref = utils.formato_cop(perm[col].mean()) if col == "ingresos" else f"{perm[col].mean():.1f}"
        texto = f"{utils.ETIQUETAS[col]}: {valor} (clientes fieles: {ref})"
        if diferencia > 0.5:
            malas.append((diferencia, texto))
        elif diferencia < -0.5:
            buenas.append((-diferencia, texto))
    ordenar = lambda lista: [t for _, t in sorted(lista, reverse=True)[:4]]
    return ordenar(malas), ordenar(buenas)


def grafica_perfil(datos: dict, df: pd.DataFrame, features: list[str]) -> go.Figure:
    """Radar: el cliente vs. promedio de los que permanecen y de los que abandonan (escala 0-100)."""
    def norm(serie_o_valor, col):
        mn, mx = df[col].min(), df[col].max()
        return ((serie_o_valor - mn) / (mx - mn) * 100).clip(0, 100) if hasattr(serie_o_valor, "clip") else \
            max(0, min(100, (serie_o_valor - mn) / (mx - mn) * 100))

    nombres = [utils.ETIQUETAS[c] for c in features]
    cliente = [norm(datos[c], c) for c in features]
    perm = [norm(df.loc[df["abandono"] == 0, c].mean(), c) for c in features]
    aban = [norm(df.loc[df["abandono"] == 1, c].mean(), c) for c in features]

    fig = go.Figure()
    for nombre, valores, color, relleno in [
        ("Promedio: permanecen", perm, VERDE, None),
        ("Promedio: abandonan", aban, ROJO, None),
        ("Este cliente", cliente, AZUL, "rgba(79,70,229,.22)"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=valores + valores[:1], theta=nombres + nombres[:1], name=nombre,
            line=dict(color=color, width=3 if nombre == "Este cliente" else 2, dash="solid" if relleno else "dot"),
            fill="toself" if relleno else None, fillcolor=relleno,
            hovertemplate="%{theta}: %{r:.0f}/100<extra>" + nombre + "</extra>",
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="#E5E7EB"),
                                 angularaxis=dict(gridcolor="#E5E7EB")))
    return estilo(fig, 340)


def grafica_medidor(prob: float, nivel: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number=dict(suffix="%", font=dict(size=40, color="#111827")),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 40, 70, 100], tickfont=dict(size=11)),
            bar=dict(color=utils.COLORES[nivel], thickness=0.28),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, 40], color="#D1FAE5"), dict(range=[40, 70], color="#FEF3C7"), dict(range=[70, 100], color="#FEE2E2")],
        ),
    ))
    return estilo(fig, 230)


def pagina_prediccion():
    modelo = get_modelo()
    df = get_datos()
    features = modelo["features"]

    cabecera("Nueva predicción", "Ingresa los datos de un cliente y el modelo estimará su probabilidad de abandono.")

    # Valores iniciales del formulario
    for k, v in dict(f_nombre="", f_edad=35, f_ingresos=3_000_000, f_frecuencia=3,
                     f_productos=8, f_tiempo=24, f_satisfaccion=6).items():
        st.session_state.setdefault(k, v)

    col_form, col_res = st.columns([1, 1.15], gap="large")

    # ------------------ FORMULARIO ------------------
    with col_form, st.container(key="panel_form"):
        titulo_panel("Datos del cliente", "Completa los campos y presiona Predecir")
        st.caption("Cargar un ejemplo:")
        b1, b2, b3 = st.columns(3)
        b1.button("🔴 Riesgo alto", on_click=cargar_ejemplo, args=("alto",), width="stretch")
        b2.button("🟡 Riesgo medio", on_click=cargar_ejemplo, args=("medio",), width="stretch")
        b3.button("🟢 Cliente fiel", on_click=cargar_ejemplo, args=("bajo",), width="stretch")

        with st.form("form_cliente", border=False):
            nombre = st.text_input("Nombre del cliente", key="f_nombre", placeholder="Ej. Laura Gómez")
            a, b = st.columns(2)
            edad = a.number_input("Edad (años)", min_value=18, max_value=100, step=1, key="f_edad")
            ingresos = b.number_input("Ingresos mensuales (COP)", min_value=0, max_value=100_000_000, step=100_000,
                                      key="f_ingresos", format="%d")
            c, d = st.columns(2)
            frecuencia = c.number_input("Frecuencia de compra (compras/mes)", min_value=0, max_value=60, step=1, key="f_frecuencia")
            productos = d.number_input("Productos adquiridos (últimos 12 meses)", min_value=0, max_value=500, step=1, key="f_productos")
            e, f = st.columns(2)
            tiempo = e.number_input("Tiempo como cliente (meses)", min_value=1, max_value=480, step=1, key="f_tiempo")
            satisfaccion = f.slider("Nivel de satisfacción (1 a 10)", min_value=1, max_value=10, key="f_satisfaccion")
            enviado = st.form_submit_button("🔮  Predecir", type="primary", width="stretch")

        if enviado:
            if not nombre.strip():
                st.error("Escribe el nombre del cliente para continuar.")
            else:
                datos = dict(edad=edad, ingresos=ingresos, frecuencia_compra=frecuencia,
                             cantidad_productos=productos, tiempo_cliente=tiempo, satisfaccion=satisfaccion)
                prob = utils.predecir(modelo, datos)
                utils.guardar_prediccion(nombre.strip(), datos, prob)
                st.session_state["ultimo"] = dict(nombre=nombre.strip(), datos=datos, prob=prob)
                st.toast("Predicción guardada en el historial", icon="✅")

    # ------------------ RESULTADO ------------------
    with col_res:
        ultimo = st.session_state.get("ultimo")
        if not ultimo:
            with st.container(key="panel_vacio"):
                vacio("🔮", "Aquí aparecerá el resultado",
                      "Completa el formulario (o carga un ejemplo) y presiona <b style='display:inline;font-size:inherit'>Predecir</b>.")
            return

        prob, nivel, datos = ultimo["prob"], utils.nivel_riesgo(ultimo["prob"]), ultimo["datos"]
        color = utils.COLORES[nivel]
        st.markdown(
            f"""
            <div class="result-card" style="--c:{color};--bg:{FONDO[nivel]}">
              <div class="result-top"><span class="badge">● RIESGO {nivel.upper()}</span>
                <span class="result-name">{html.escape(ultimo['nombre'])}</span></div>
              <div class="result-title">Riesgo {nivel.lower()} de abandono</div>
              <div class="result-prob"><span>{prob:.0%}</span>probabilidad estimada</div>
              <div class="bar"><div class="bar-fill" style="width:{prob * 100:.1f}%"></div></div>
              <div class="bar-scale"><span style="width:40%">Bajo</span><span style="width:30%">Medio</span><span style="width:30%">Alto</span></div>
              <div class="result-reco"><b>Acción sugerida:</b> {RECOMENDACION[nivel]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        malas, buenas = senales(datos, df, features)
        if malas or buenas:
            st.write("")
            with st.container(key="panel_senales"):
                titulo_panel("Señales frente a los clientes que permanecen", "Comparado con el promedio de los clientes fieles de la base")
                chips = "".join(f'<span class="chip chip-mal">⚠ {t}</span>' for t in malas)
                chips += "".join(f'<span class="chip chip-bien">✓ {t}</span>' for t in buenas)
                st.markdown(chips, unsafe_allow_html=True)

    # Gráficas del resultado (ancho completo, debajo)
    if st.session_state.get("ultimo"):
        st.write("")
        g1, g2 = st.columns(2)
        with g1, st.container(key="panel_medidor"):
            titulo_panel("Indicador de riesgo", "Probabilidad de abandono estimada por el modelo")
            mostrar(grafica_medidor(prob, nivel))
        with g2, st.container(key="panel_radar"):
            titulo_panel("Perfil del cliente", "Cada variable en escala 0–100 respecto a toda la base")
            mostrar(grafica_perfil(datos, df, features))


# ---------------------------------------------------------------------------
# PÁGINA 3: HISTORIAL
# ---------------------------------------------------------------------------
@st.dialog("Eliminar historial")
def confirmar_borrado():
    st.write("Se borrarán **todas** las predicciones guardadas. Esta acción no se puede deshacer.")
    a, b = st.columns(2)
    if a.button("Cancelar", width="stretch"):
        st.rerun()
    if b.button("Sí, eliminar", type="primary", width="stretch"):
        utils.borrar_historial()
        st.session_state.pop("ultimo", None)
        st.rerun()


def pagina_historial():
    cabecera("Historial de predicciones", "Cada predicción realizada queda registrada con sus datos, resultado y probabilidad.")
    hist = utils.leer_historial()

    if hist.empty:
        with st.container(key="panel_hvacio"):
            vacio("🗂️", "El historial está vacío", "Las predicciones que hagas aparecerán aquí automáticamente.")
        return

    tarjetas([
        ("list", "#EEF2FF", AZUL, "Predicciones registradas", f"{len(hist)}", "Total en el historial"),
        ("alert", "#FEF2F2", ROJO, "Con riesgo alto", f"{int((hist['riesgo'] == 'Alto').sum())}", "Requieren acción"),
        ("activity", "#FFFBEB", AMBAR, "Probabilidad promedio", f"{hist['probabilidad'].mean():.1%}", "De los clientes analizados"),
    ])

    with st.container(key="panel_historial"):
        f1, f2 = st.columns([2, 3])
        buscar = f1.text_input("Buscar por nombre", placeholder="Escribe un nombre…")
        niveles = f2.multiselect("Nivel de riesgo", ["Bajo", "Medio", "Alto"], default=["Bajo", "Medio", "Alto"])

        vista = hist[hist["riesgo"].isin(niveles)]
        if buscar:
            vista = vista[vista["nombre"].str.contains(buscar, case=False, na=False)]

        tabla = pd.DataFrame({
            "Nombre": vista["nombre"],
            "Fecha": vista["fecha"],
            "Edad": vista["edad"],
            "Ingresos": vista["ingresos"].map(utils.formato_cop),
            "Compras/mes": vista["frecuencia_compra"],
            "Productos": vista["cantidad_productos"],
            "Meses": vista["tiempo_cliente"],
            "Satisfacción": vista["satisfaccion"],
            "Resultado": vista["riesgo"].map(lambda r: f"{utils.ICONOS[r]} Riesgo {r.lower()}"),
            "Probabilidad": vista["probabilidad"] * 100,
        })
        st.dataframe(
            tabla, hide_index=True, width="stretch",
            column_config={"Probabilidad": st.column_config.ProgressColumn("Probabilidad", format="%.0f%%", min_value=0, max_value=100)},
        )
        st.caption(f"Mostrando {len(vista)} de {len(hist)} predicciones")

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("⬇️ Descargar CSV", data=hist.to_csv(index=False).encode("utf-8"),
                           file_name="historial_predicciones.csv", mime="text/csv", width="stretch")
        if d2.button("🗑️ Eliminar historial", width="stretch"):
            confirmar_borrado()


# ---------------------------------------------------------------------------
# PÁGINA 4: ANÁLISIS DEL MODELO
# ---------------------------------------------------------------------------
def pagina_modelo():
    m = get_metricas()
    res = m["resultados"]
    mejor = m["modelo_seleccionado"]

    cabecera("Análisis del modelo", "Comparación de los modelos de clasificación entrenados y el que usa la aplicación.")

    tarjetas([
        ("cpu", "#EEF2FF", AZUL, "Modelo en uso", mejor, "Seleccionado por mayor accuracy"),
        ("target", "#ECFDF5", VERDE, "Accuracy (prueba)", f"{res[mejor]['accuracy']:.1%}", f"ROC-AUC {res[mejor]['roc_auc']:.2f}"),
        ("database", "#FFFBEB", AMBAR, "Datos de entrenamiento", f"{m['n_train']}", f"{m['n_test']} para prueba · {m['n_total']} en total"),
    ])

    # ---- Comparación ----
    with st.container(key="panel_comparacion"):
        titulo_panel("Comparación de modelos", f"Criterio de selección: {m['criterio'].lower()}")
        cmp = pd.DataFrame({
            "Modelo": [("⭐ " if k == mejor else "") + k for k in res],
            "Accuracy": [res[k]["accuracy"] for k in res],
            "Precisión (Abandona)": [res[k]["precision"] for k in res],
            "Recall (Abandona)": [res[k]["recall"] for k in res],
            "F1 (Abandona)": [res[k]["f1"] for k in res],
            "ROC-AUC": [res[k]["roc_auc"] for k in res],
            "Accuracy validación cruzada": [res[k]["cv_accuracy_media"] for k in res],
        })
        izq, der = st.columns([1.1, 1])
        with izq:
            st.dataframe(cmp, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in cmp.columns[1:]})
        with der:
            largo = cmp.melt(id_vars="Modelo", value_vars=["Accuracy", "F1 (Abandona)", "ROC-AUC"], var_name="Métrica", value_name="Valor")
            fig = px.bar(largo, x="Modelo", y="Valor", color="Métrica", barmode="group",
                         color_discrete_sequence=["#6366F1", "#F43F5E", "#10B981"], labels={"Modelo": ""})
            fig.update_traces(marker_line_width=0)
            fig.update_layout(yaxis_range=[0, 1], bargap=0.25)
            mostrar(estilo(fig, 290))

    st.write("")
    # ---- Matriz de confusión + reporte por modelo ----
    with st.container(key="panel_detalle"):
        titulo_panel("Matriz de confusión y reporte de clasificación", "Evaluación sobre el conjunto de prueba (datos que el modelo no vio al entrenar)")
        pestanas = st.tabs([("⭐ " if k == mejor else "") + k for k in res])
        for pestana, nombre in zip(pestanas, res):
            with pestana:
                x, y = st.columns([1, 1.1])
                with x:
                    cm = res[nombre]["matriz_confusion"]
                    fig = px.imshow(cm, text_auto=True, aspect="auto",
                                    x=["Predicho: Permanece", "Predicho: Abandona"], y=["Real: Permanece", "Real: Abandona"],
                                    color_continuous_scale=[[0, "#EEF2FF"], [1, "#4F46E5"]])
                    fig.update_traces(textfont_size=24)
                    fig.update_layout(coloraxis_showscale=False)
                    mostrar(estilo(fig, 300))
                with y:
                    rep = res[nombre]["reporte"]
                    filas = {k: rep[k] for k in ["Permanece", "Abandona", "macro avg", "weighted avg"]}
                    tabla = pd.DataFrame(filas).T.rename(
                        columns={"precision": "Precisión", "recall": "Recall", "f1-score": "F1", "support": "Casos"})
                    tabla["Casos"] = tabla["Casos"].astype(int)
                    st.dataframe(tabla, width="stretch",
                                 column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["Precisión", "Recall", "F1"]})
                    st.caption(f"Accuracy global: **{res[nombre]['accuracy']:.2%}**")

    st.write("")
    c1, c2 = st.columns([1.2, 1])
    with c1, st.container(key="panel_importancia"):
        titulo_panel("Variables más importantes", f"Qué tanto empeora el accuracy de «{mejor}» al desordenar cada variable")
        imp = pd.DataFrame({"Variable": [utils.ETIQUETAS[k] for k in m["importancias"]], "Importancia": list(m["importancias"].values())})
        fig = px.bar(imp.sort_values("Importancia"), x="Importancia", y="Variable", orientation="h", color_discrete_sequence=["#6366F1"])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.35, yaxis_title=None)
        mostrar(estilo(fig, 300))
    with c2, st.container(key="panel_proceso"):
        titulo_panel("Proceso seguido")
        st.markdown(
            """
1. **Dataset:** 500 clientes con 6 variables + abandono (sí/no).
2. **Limpieza (Pandas):** duplicados, valores imposibles y nulos (mediana).
3. **X e y:** 6 variables de entrada y `abandono` como objetivo.
4. **Train/Test:** 80 % / 20 % estratificado.
5. **Modelos:** Regresión Logística, Random Forest y KNN.
6. **Evaluación:** accuracy, matriz de confusión, reporte y validación cruzada.
7. **Selección:** el mejor se guarda en `ml/modelo.pkl` y lo usa esta app.
            """
        )


# ---------------------------------------------------------------------------
# NAVEGACIÓN
# ---------------------------------------------------------------------------
paginas = st.navigation([
    st.Page(pagina_dashboard, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    st.Page(pagina_prediccion, title="Nueva predicción", icon=":material/auto_awesome:", url_path="prediccion"),
    st.Page(pagina_historial, title="Historial", icon=":material/history:", url_path="historial"),
    st.Page(pagina_modelo, title="Análisis del modelo", icon=":material/analytics:", url_path="modelo"),
])

_m = get_metricas()
st.sidebar.markdown(
    f'<div class="side-card"><b>Modelo activo</b><br>{_m["modelo_seleccionado"]}<br>'
    f'Accuracy: {_m["resultados"][_m["modelo_seleccionado"]]["accuracy"]:.1%}<br>'
    f'Clientes en la base: {_m["n_total"]}</div>',
    unsafe_allow_html=True,
)

paginas.run()
