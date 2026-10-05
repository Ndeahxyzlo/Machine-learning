import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import utils

BASE = Path(__file__).parent

st.set_page_config(
    page_title="ChurnGuard | Predicción de abandono",
    page_icon=str(BASE / "assets" / "icono.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)

TINTA = "#1C2430"
MUTED = "#5B6572"
BORDE = "#D5D9DF"
ACENTO = "#1F4E79"
BAJO, MEDIO, ALTO = utils.COLORES["Bajo"], utils.COLORES["Medio"], utils.COLORES["Alto"]
ESTADO_COLORES = {"Permanece": "#4A6785", "Abandona": "#C26A1C"}
FUENTE = "IBM Plex Sans, Segoe UI, Arial, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --tinta: #1C2430; --muted: #5B6572; --tenue: #8A939E; --borde: #D5D9DF;
    --acento: #1F4E79; --bajo: #2F7D5B; --medio: #A8741A; --alto: #B3362D;
}

.stApp, .stApp button, .stApp input, .stApp textarea, .stApp [data-baseweb] {
    font-family: 'IBM Plex Sans', 'Segoe UI', Arial, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }
footer { visibility: hidden; }

.stApp button, .stApp [data-baseweb], .stApp [data-baseweb] *,
.stApp [data-testid="stAlert"], .stApp [data-testid="stSidebar"] a,
.stApp [data-testid="stDialog"] div[role="dialog"] { border-radius: 0 !important; }
[data-baseweb="popover"] > div, [data-testid="stDialog"] div[role="dialog"],
[data-testid="stToast"], [data-testid="stNotification"] {
    box-shadow: none !important; border: 1px solid var(--borde);
}

.page-head { border-bottom: 1px solid var(--borde); margin: 0 0 1.4rem; padding-bottom: .9rem; }
.page-head h1 { font-size: 1.65rem; font-weight: 600; color: var(--tinta); margin: 0; padding: 0; letter-spacing: -.01em; }
.page-head p { color: var(--muted); margin: .3rem 0 0; font-size: .95rem; }

.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 0 0 20px; }
.kpi { background: #fff; border: 1px solid var(--borde); border-left: 4px solid var(--c, var(--acento)); padding: 14px 18px; }
.kpi-label { font-size: .72rem; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }
.kpi-value { font-size: 1.9rem; font-weight: 600; color: var(--tinta); line-height: 1.2; margin-top: 4px; font-variant-numeric: tabular-nums; }
.kpi-value.corto { font-size: 1.45rem; padding: .35rem 0 .15rem; }
.kpi-sub { font-size: .78rem; color: var(--tenue); margin-top: 2px; }

[class*="st-key-panel"] { background: #fff; border: 1px solid var(--borde); padding: 1.1rem 1.3rem 1rem; }
.panel-title { font-weight: 600; font-size: .95rem; color: var(--tinta); margin: 0; }
.panel-sub { font-size: .8rem; color: var(--tenue); margin: 0 0 .5rem; }

.stApp label p { font-weight: 500; font-size: .84rem; color: #2B3440; }
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
    background: var(--acento); border: 1px solid var(--acento); color: #fff;
    font-weight: 500; padding: .6rem 1rem; box-shadow: none;
}
[data-testid="stBaseButton-primaryFormSubmit"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: #173B5C; border-color: #173B5C; color: #fff;
}
[data-testid="stBaseButton-secondary"] { background: #fff; border: 1px solid #AEB6C0; color: var(--tinta); font-weight: 500; box-shadow: none; padding: .4rem .5rem; }
[data-testid="stBaseButton-secondary"] p { font-size: .84rem; white-space: nowrap; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--acento); color: var(--acento); }

.result-card { background: #fff; border: 1px solid var(--borde); border-left: 5px solid var(--c); padding: 20px 24px; }
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.tag { font-size: .7rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--c); border: 1px solid var(--c); padding: .2rem .55rem; }
.result-name { font-weight: 500; color: var(--muted); font-size: .92rem; }
.result-title { font-size: 1.4rem; font-weight: 600; color: var(--tinta); margin: 14px 0 2px; }
.result-prob { color: var(--muted); font-size: .9rem; }
.result-prob span { font-size: 3rem; font-weight: 600; color: var(--c); margin-right: 8px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
.bar { position: relative; height: 12px; background: #E6E9ED; margin: 16px 0 4px; }
.bar-fill { height: 100%; background: var(--c); }
.bar-tick { position: absolute; top: -4px; bottom: -4px; width: 1px; background: #6B7480; }
.bar-scale { position: relative; height: 1.1rem; font-size: .72rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.bar-scale span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.bar-scale .ini { transform: none; }
.bar-scale .fin { transform: translateX(-100%); }
.bar-bands { position: relative; height: 1.1rem; font-size: .7rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--tenue); }
.bar-bands span { position: absolute; transform: translateX(-50%); }
.result-reco { margin: 16px 0 0; padding-top: 12px; border-top: 1px solid var(--borde); color: #2B3440; font-size: .9rem; }

.tabla-wrap { overflow-x: auto; }
.tabla { width: 100%; border-collapse: collapse; font-size: .82rem; }
.tabla th { text-align: left; font-weight: 500; color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid var(--borde); padding: .45rem .4rem; }
.tabla td { padding: .5rem .4rem; border-bottom: 1px solid #ECEEF1; color: var(--tinta); font-variant-numeric: tabular-nums; }
.tabla th.num, .tabla td.num { text-align: right; }
.tabla tr:last-child td { border-bottom: 0; }
.tabla td.lect-mal { color: var(--alto); font-weight: 600; }
.tabla td.lect-bien { color: var(--bajo); font-weight: 600; }
.tabla td.lect-neu { color: var(--tenue); }

.vacio { text-align: center; padding: 2.4rem 1rem; color: var(--muted); font-size: .92rem; }
.vacio b { display: block; color: var(--tinta); font-size: 1rem; font-weight: 600; margin-bottom: .25rem; }

.side-card { border: 1px solid rgba(255,255,255,.16); padding: 12px 14px; font-size: .78rem; line-height: 1.7; color: #C3CCD6; }
.side-card .t { color: #fff; font-weight: 600; font-size: .8rem; margin-bottom: 2px; }
.side-card .f { display: flex; justify-content: space-between; gap: 8px; }
.side-card .f span:last-child { color: #fff; font-variant-numeric: tabular-nums; text-align: right; }

@media (max-width: 640px) {
    .block-container { padding: 3.4rem .8rem 3rem; }
    .page-head h1 { font-size: 1.4rem; }
    .kpi-value { font-size: 1.55rem; }
    .result-prob span { font-size: 2.4rem; }
    .result-title { font-size: 1.2rem; }
    [class*="st-key-panel"] { padding: .9rem .9rem .8rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.logo(str(BASE / "assets" / "logo.svg"), icon_image=str(BASE / "assets" / "icono.svg"), size="large")

RECOMENDACION = {
    "Alto": "Contactar al cliente en las próximas 48 horas con una oferta de retención personalizada y revisar su experiencia reciente.",
    "Medio": "Hacer seguimiento con una encuesta de satisfacción y ofrecer beneficios de fidelización antes de que el riesgo aumente.",
    "Bajo": "Cliente estable. Mantener la comunicación regular y considerar programas de recompensa para reforzar la relación.",
}


@st.cache_resource
def get_modelo():
    return utils.cargar_modelo()


@st.cache_data
def get_metricas():
    return utils.cargar_metricas()


@st.cache_data
def get_datos():
    df = utils.puntuar_dataset(get_modelo(), utils.cargar_datos())
    df["estado"] = df["abandono"].map({0: "Permanece", 1: "Abandona"})
    return df


utils.init_db()


def cabecera(titulo: str, subtitulo: str) -> None:
    st.markdown(f'<div class="page-head"><h1>{titulo}</h1><p>{subtitulo}</p></div>', unsafe_allow_html=True)


def tarjetas(items: list[tuple]) -> None:
    cuerpo = ""
    for color, etiqueta, valor, sub in items:
        cuerpo += (
            f'<div class="kpi" style="--c:{color}"><div class="kpi-label">{etiqueta}</div>'
            f'<div class="kpi-value{" corto" if len(str(valor)) > 11 else ""}">{valor}</div><div class="kpi-sub">{sub}</div></div>'
        )
    st.markdown(f'<div class="kpi-grid">{cuerpo}</div>', unsafe_allow_html=True)


def titulo_panel(titulo: str, sub: str = "") -> None:
    st.markdown(f'<p class="panel-title">{titulo}</p><p class="panel-sub">{sub}</p>', unsafe_allow_html=True)


def estilo(fig: go.Figure, alto: int = 300, leyenda: bool = False) -> go.Figure:
    fig.update_layout(
        height=alto,
        margin=dict(l=4, r=8, t=34 if leyenda else 10, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FUENTE, size=12.5, color="#2B3440"),
        showlegend=leyenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, xanchor="left", title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDE, font=dict(family=FUENTE, color=TINTA)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#AEB6C0", ticks="outside", tickcolor="#AEB6C0")
    fig.update_yaxes(gridcolor="#E8EBEF", zeroline=False, showline=False)
    return fig


def mostrar(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def vacio(titulo: str, texto: str) -> None:
    st.markdown(f'<div class="vacio"><b>{titulo}</b>{texto}</div>', unsafe_allow_html=True)


def pagina_dashboard():
    datos = get_datos()
    historial = utils.leer_historial()

    cabecera("Dashboard", "Resumen del riesgo de abandono en la base de clientes.")

    total = len(datos)
    n_alto = int((datos["riesgo"] == "Alto").sum())
    n_bajo = int((datos["riesgo"] == "Bajo").sum())
    promedio = datos["probabilidad"].mean()

    tarjetas([
        (ACENTO, "Clientes analizados", f"{total:,}".replace(",", "."), f"{len(historial)} predicciones nuevas en el historial"),
        (ALTO, "Riesgo alto", f"{n_alto}", f"{n_alto / total:.1%} de la base"),
        (BAJO, "Riesgo bajo", f"{n_bajo}", f"{n_bajo / total:.1%} de la base"),
        (MEDIO, "Prob. promedio de abandono", f"{promedio:.1%}", "Estimada por el modelo"),
    ])

    c1, c2 = st.columns(2)
    with c1, st.container(key="panel_riesgo"):
        titulo_panel("Clientes por nivel de riesgo", "Bajo: menos de 40 %. Medio: de 40 a 70 %. Alto: 70 % o más.")
        conteo = datos["riesgo"].value_counts().reindex(["Bajo", "Medio", "Alto"]).fillna(0).astype(int).reset_index()
        conteo.columns = ["riesgo", "clientes"]
        conteo["texto"] = conteo["clientes"].map(lambda n: f"{n} ({n / total:.0%})")
        fig = px.bar(conteo, x="riesgo", y="clientes", color="riesgo", text="texto", color_discrete_map=utils.COLORES,
                     labels={"riesgo": "", "clientes": "Clientes"})
        fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
        fig.update_layout(bargap=0.5, yaxis_range=[0, conteo["clientes"].max() * 1.18])
        mostrar(estilo(fig))

    with c2, st.container(key="panel_satisfaccion"):
        titulo_panel("Nivel de satisfacción", "Clientes que abandonaron frente a los que permanecieron")
        fig = px.histogram(datos, x="satisfaccion", color="estado", barmode="group", nbins=10,
                           color_discrete_map=ESTADO_COLORES, category_orders={"estado": ["Permanece", "Abandona"]},
                           labels={"satisfaccion": "Satisfacción (1 a 10)"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.25, yaxis_title="Clientes")
        fig.update_xaxes(dtick=1)
        mostrar(estilo(fig, leyenda=True))

    st.write("")
    c3, c4 = st.columns(2)
    with c3, st.container(key="panel_edades"):
        titulo_panel("Distribución de edades", "Cantidad de clientes por rango de edad")
        fig = px.histogram(datos, x="edad", nbins=14, color_discrete_sequence=[ACENTO], labels={"edad": "Edad (años)"})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.08, yaxis_title="Clientes")
        mostrar(estilo(fig))

    with c4, st.container(key="panel_antiguedad"):
        titulo_panel("Abandono según antigüedad", "Porcentaje de clientes que abandonan por tiempo en la empresa")
        tramos = pd.cut(datos["tiempo_cliente"], bins=[0, 6, 12, 24, 48, 1000],
                        labels=["0 a 6 meses", "6 a 12 meses", "1 a 2 años", "2 a 4 años", "Más de 4 años"])
        tasa = (datos.groupby(tramos, observed=True)["abandono"].mean() * 100).round(1).reset_index()
        tasa.columns = ["tramo", "tasa"]
        fig = px.bar(tasa, x="tramo", y="tasa", text="tasa", color_discrete_sequence=[ACENTO],
                     labels={"tramo": "", "tasa": "% de abandono"})
        fig.update_traces(texttemplate="%{text:.0f} %", textposition="outside", marker_line_width=0, cliponaxis=False)
        fig.update_layout(bargap=0.4, yaxis_range=[0, max(tasa["tasa"]) * 1.2])
        mostrar(estilo(fig))

    st.write("")
    with st.container(key="panel_ultimas"):
        titulo_panel("Últimas predicciones realizadas", "Las 5 más recientes del historial")
        if historial.empty:
            vacio("Aún no hay predicciones", "Ve a Nueva predicción para analizar el primer cliente.")
        else:
            tabla = historial.head(5).copy()
            tabla["Riesgo"] = tabla["riesgo"]
            tabla["Probabilidad"] = tabla["probabilidad"] * 100
            st.dataframe(
                tabla[["nombre", "fecha", "satisfaccion", "Riesgo", "Probabilidad"]].rename(
                    columns={"nombre": "Cliente", "fecha": "Fecha", "satisfaccion": "Satisfacción"}),
                hide_index=True, width="stretch",
                column_config={"Probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad de abandono", format="%.0f %%", min_value=0, max_value=100)},
            )


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


def formatear(col: str, valor: float, decimales: int = 1) -> str:
    if col == "ingresos":
        return utils.formato_cop(valor)
    if decimales == 0:
        return f"{valor:g}"
    return f"{valor:.{decimales}f}"


def comparacion(datos: dict, df: pd.DataFrame, features: list[str]) -> list[dict]:
    perm, aban = df[df["abandono"] == 0], df[df["abandono"] == 1]
    filas = []
    for col in features:
        desv = df[col].std()
        sentido = 1 if aban[col].mean() > perm[col].mean() else -1
        relevante = abs(aban[col].mean() - perm[col].mean()) / desv >= 0.15
        diferencia = (datos[col] - perm[col].mean()) / desv * sentido
        if relevante and diferencia > 0.5:
            lectura, clase = "Desfavorable", "lect-mal"
        elif relevante and diferencia < -0.5:
            lectura, clase = "Favorable", "lect-bien"
        else:
            lectura, clase = "Neutral", "lect-neu"
        filas.append(dict(
            variable=utils.ETIQUETAS[col],
            cliente=formatear(col, datos[col], 0),
            perm=formatear(col, perm[col].mean()),
            aban=formatear(col, aban[col].mean()),
            lectura=lectura, clase=clase,
        ))
    return filas


def tabla_comparacion(filas: list[dict]) -> str:
    cuerpo = "".join(
        f'<tr><td>{f["variable"]}</td><td class="num">{f["cliente"]}</td><td class="num">{f["perm"]}</td>'
        f'<td class="num">{f["aban"]}</td><td class="{f["clase"]}">{f["lectura"]}</td></tr>'
        for f in filas
    )
    return (
        '<div class="tabla-wrap"><table class="tabla"><thead><tr><th>Variable</th><th class="num">Cliente</th>'
        '<th class="num">Permanecen</th><th class="num">Abandonan</th>'
        f'<th>Lectura</th></tr></thead><tbody>{cuerpo}</tbody></table></div>'
    )


def grafica_perfil(datos: dict, df: pd.DataFrame, features: list[str]) -> go.Figure:
    def norm(valor, col):
        mn, mx = df[col].min(), df[col].max()
        return max(0.0, min(100.0, (valor - mn) / (mx - mn) * 100))

    nombres = [utils.ETIQUETAS[c] for c in features]
    cliente = [norm(datos[c], c) for c in features]
    perm = [norm(df.loc[df["abandono"] == 0, c].mean(), c) for c in features]
    aban = [norm(df.loc[df["abandono"] == 1, c].mean(), c) for c in features]

    fig = go.Figure()
    xs, ys = [], []
    for n, p, a in zip(nombres, perm, aban):
        xs += [p, a, None]
        ys += [n, n, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#C9CFD6", width=3),
                             hoverinfo="skip", showlegend=False))
    for etiqueta, valores, simbolo, color, tam in [
        ("Permanecen", perm, "circle", ESTADO_COLORES["Permanece"], 11),
        ("Abandonan", aban, "diamond", ESTADO_COLORES["Abandona"], 12),
        ("Este cliente", cliente, "square", TINTA, 13),
    ]:
        fig.add_trace(go.Scatter(
            x=valores, y=nombres, mode="markers", name=etiqueta,
            marker=dict(symbol=simbolo, size=tam, color=color, line=dict(color="white", width=1.5)),
            hovertemplate="%{y}: %{x:.0f} de 100<extra>" + etiqueta + "</extra>",
        ))
    fig.update_layout(xaxis=dict(range=[-3, 103], tickvals=[0, 25, 50, 75, 100],
                                 title="0 = mínimo de la base, 100 = máximo",
                                 showgrid=True, gridcolor="#E8EBEF"),
                      yaxis=dict(autorange="reversed", showgrid=False))
    return estilo(fig, 340, leyenda=True)


def grafica_efectos(efectos: dict) -> go.Figure:
    filas = sorted(((utils.ETIQUETAS[k], v * 100) for k, v in efectos.items()), key=lambda t: t[1])
    nombres = [f[0] for f in filas]
    valores = [f[1] for f in filas]
    colores = [ALTO if v > 0 else BAJO for v in valores]
    fig = go.Figure(go.Bar(
        x=valores, y=nombres, orientation="h", marker=dict(color=colores, line_width=0),
        text=[f"{v:+.1f} pp" for v in valores], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:+.1f} puntos porcentuales<extra></extra>",
    ))
    tope = max(abs(v) for v in valores) * 1.9 or 1
    fig.update_layout(bargap=0.45, xaxis=dict(range=[-tope, tope], zeroline=True, zerolinecolor="#6B7480",
                                              title="Puntos porcentuales",
                                              showgrid=True, gridcolor="#E8EBEF"),
                      yaxis=dict(showgrid=False))
    return estilo(fig, 340)


def barra_resultado(prob: float, color: str) -> str:
    marcas = "".join(f'<div class="bar-tick" style="left:{p}%"></div>' for p in (40, 70))
    escala = (
        '<span class="ini" style="left:0">0 %</span><span style="left:40%">40 %</span>'
        '<span style="left:70%">70 %</span><span class="fin" style="left:100%">100 %</span>'
    )
    bandas = '<span style="left:20%">Bajo</span><span style="left:55%">Medio</span><span style="left:85%">Alto</span>'
    return (
        f'<div class="bar"><div class="bar-fill" style="width:{prob * 100:.1f}%"></div>{marcas}</div>'
        f'<div class="bar-scale">{escala}</div><div class="bar-bands">{bandas}</div>'
    )


def pagina_prediccion():
    modelo = get_modelo()
    df = get_datos()
    features = modelo["features"]

    cabecera("Nueva predicción", "Ingresa los datos de un cliente y el modelo estimará su probabilidad de abandono.")

    for k, v in dict(f_nombre="", f_edad=35, f_ingresos=3_000_000, f_frecuencia=3,
                     f_productos=8, f_tiempo=24, f_satisfaccion=6).items():
        st.session_state.setdefault(k, v)

    col_form, col_res = st.columns([1, 1.15], gap="large")

    with col_form, st.container(key="panel_form"):
        titulo_panel("Datos del cliente", "Completa los campos y presiona Predecir")
        st.caption("Cargar un ejemplo")
        b1, b2, b3 = st.columns(3)
        b1.button("Riesgo alto", on_click=cargar_ejemplo, args=("alto",), width="stretch")
        b2.button("Riesgo medio", on_click=cargar_ejemplo, args=("medio",), width="stretch")
        b3.button("Cliente fiel", on_click=cargar_ejemplo, args=("bajo",), width="stretch")

        with st.form("form_cliente", border=False):
            nombre = st.text_input("Nombre del cliente", key="f_nombre", placeholder="Ej. Laura Gómez")
            a, b = st.columns(2)
            edad = a.number_input("Edad (años)", min_value=18, max_value=100, step=1, key="f_edad")
            ingresos = b.number_input("Ingresos mensuales (COP)", min_value=0, max_value=100_000_000, step=100_000,
                                      key="f_ingresos", format="%d")
            c, d = st.columns(2)
            frecuencia = c.number_input("Frecuencia de compra (compras por mes)", min_value=0, max_value=60, step=1, key="f_frecuencia")
            productos = d.number_input("Productos adquiridos (últimos 12 meses)", min_value=0, max_value=500, step=1, key="f_productos")
            e, f = st.columns(2)
            tiempo = e.number_input("Tiempo como cliente (meses)", min_value=1, max_value=480, step=1, key="f_tiempo")
            satisfaccion = f.slider("Nivel de satisfacción (1 a 10)", min_value=1, max_value=10, key="f_satisfaccion")
            enviado = st.form_submit_button("Predecir", type="primary", width="stretch")

        if enviado:
            if not nombre.strip():
                st.error("Escribe el nombre del cliente para continuar.")
            else:
                datos = dict(edad=edad, ingresos=ingresos, frecuencia_compra=frecuencia,
                             cantidad_productos=productos, tiempo_cliente=tiempo, satisfaccion=satisfaccion)
                prob = utils.predecir(modelo, datos)
                utils.guardar_prediccion(nombre.strip(), datos, prob)
                st.session_state["ultimo"] = dict(nombre=nombre.strip(), datos=datos, prob=prob)
                st.toast("Predicción guardada en el historial")

    ultimo = st.session_state.get("ultimo")

    with col_res:
        if not ultimo:
            with st.container(key="panel_vacio"):
                vacio("El resultado aparecerá aquí",
                      "Completa el formulario o carga un ejemplo y presiona Predecir.")
        else:
            prob, nivel = ultimo["prob"], utils.nivel_riesgo(ultimo["prob"])
            color = utils.COLORES[nivel]
            st.markdown(
                f"""
                <div class="result-card" style="--c:{color}">
                  <div class="result-top"><span class="tag">Riesgo {nivel.lower()}</span>
                    <span class="result-name">{html.escape(ultimo['nombre'])}</span></div>
                  <div class="result-title">Riesgo {nivel.lower()} de abandono</div>
                  <div class="result-prob"><span>{prob:.0%}</span>probabilidad estimada</div>
                  {barra_resultado(prob, color)}
                  <div class="result-reco"><b>Acción sugerida.</b> {RECOMENDACION[nivel]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            with st.container(key="panel_comparacion_cliente"):
                titulo_panel("Comparación con la base de clientes",
                             "Valor del cliente frente al promedio de quienes permanecen y de quienes abandonan")
                st.markdown(tabla_comparacion(comparacion(ultimo["datos"], df, features)), unsafe_allow_html=True)

    if ultimo:
        st.write("")
        g1, g2 = st.columns(2)
        with g1, st.container(key="panel_efectos"):
            titulo_panel("Efecto de cada variable en esta predicción",
                         "Cambio en la probabilidad si la variable tomara el valor típico de la base. Positivo: aumenta el riesgo. Negativo: lo reduce.")
            mostrar(grafica_efectos(utils.efecto_variables(modelo, ultimo["datos"], df)))
        with g2, st.container(key="panel_perfil"):
            titulo_panel("Perfil del cliente", "Posición de cada variable dentro del rango de la base. Los marcadores redondo y rombo son los promedios de cada grupo.")
            mostrar(grafica_perfil(ultimo["datos"], df, features))


@st.dialog("Eliminar historial")
def confirmar_borrado():
    st.write("Se borrarán todas las predicciones guardadas. Esta acción no se puede deshacer.")
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
            vacio("El historial está vacío", "Las predicciones que hagas aparecerán aquí automáticamente.")
        return

    tarjetas([
        (ACENTO, "Predicciones registradas", f"{len(hist)}", "Total en el historial"),
        (ALTO, "Con riesgo alto", f"{int((hist['riesgo'] == 'Alto').sum())}", "Requieren acción"),
        (MEDIO, "Probabilidad promedio", f"{hist['probabilidad'].mean():.1%}", "De los clientes analizados"),
    ])

    with st.container(key="panel_historial"):
        f1, f2 = st.columns([2, 3])
        buscar = f1.text_input("Buscar por nombre", placeholder="Escribe un nombre")
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
            "Resultado": vista["riesgo"].map(lambda r: f"Riesgo {r.lower()}"),
            "Probabilidad": vista["probabilidad"] * 100,
        })
        st.dataframe(
            tabla, hide_index=True, width="stretch",
            column_config={"Probabilidad": st.column_config.ProgressColumn("Probabilidad", format="%.0f %%", min_value=0, max_value=100)},
        )
        st.caption(f"Mostrando {len(vista)} de {len(hist)} predicciones")

        d1, d2, _ = st.columns([1, 1, 2])
        d1.download_button("Descargar CSV", data=hist.to_csv(index=False).encode("utf-8"),
                           file_name="historial_predicciones.csv", mime="text/csv", width="stretch")
        if d2.button("Eliminar historial", width="stretch"):
            confirmar_borrado()


def pagina_modelo():
    m = get_metricas()
    res = m["resultados"]
    mejor = m["modelo_seleccionado"]

    cabecera("Análisis del modelo", "Comparación de los modelos de clasificación entrenados y el que usa la aplicación.")

    tarjetas([
        (ACENTO, "Modelo en uso", mejor, "Seleccionado por mayor accuracy"),
        (BAJO, "Accuracy en prueba", f"{res[mejor]['accuracy']:.1%}", f"ROC-AUC {res[mejor]['roc_auc']:.2f}"),
        (MEDIO, "Datos de entrenamiento", f"{m['n_train']}", f"{m['n_test']} para prueba, {m['n_total']} en total"),
    ])

    def nombre_modelo(k: str) -> str:
        return f"{k} (en uso)" if k == mejor else k

    with st.container(key="panel_comparacion"):
        titulo_panel("Comparación de modelos", f"Criterio de selección: {m['criterio']}")
        cmp = pd.DataFrame({
            "Modelo": [nombre_modelo(k) for k in res],
            "Accuracy": [res[k]["accuracy"] for k in res],
            "Precisión (Abandona)": [res[k]["precision"] for k in res],
            "Recall (Abandona)": [res[k]["recall"] for k in res],
            "F1 (Abandona)": [res[k]["f1"] for k in res],
            "ROC-AUC": [res[k]["roc_auc"] for k in res],
            "Accuracy val. cruzada": [res[k]["cv_accuracy_media"] for k in res],
        })
        st.dataframe(cmp, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in cmp.columns[1:]})
        largo = cmp.assign(Modelo=list(res)).melt(id_vars="Modelo", value_vars=["Accuracy", "F1 (Abandona)", "ROC-AUC"],
                                                  var_name="Métrica", value_name="Valor")
        fig = px.bar(largo, x="Modelo", y="Valor", color="Métrica", barmode="group",
                     color_discrete_sequence=[ACENTO, "#C26A1C", "#7C8A99"], labels={"Modelo": ""})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(yaxis_range=[0, 1], bargap=0.5)
        mostrar(estilo(fig, 300, leyenda=True))

    st.write("")
    with st.container(key="panel_detalle"):
        titulo_panel("Matriz de confusión y reporte de clasificación",
                     "Evaluación sobre el conjunto de prueba, con datos que el modelo no vio al entrenar")
        pestanas = st.tabs([nombre_modelo(k) for k in res])
        for pestana, nombre in zip(pestanas, res):
            with pestana:
                x, y = st.columns([1, 1.1])
                with x:
                    cm = res[nombre]["matriz_confusion"]
                    fig = px.imshow(cm, text_auto=True, aspect="auto",
                                    x=["Predicho: Permanece", "Predicho: Abandona"], y=["Real: Permanece", "Real: Abandona"],
                                    color_continuous_scale=[[0, "#F1F4F8"], [1, ACENTO]])
                    fig.update_traces(textfont_size=22, xgap=3, ygap=3)
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_xaxes(side="bottom", ticks="")
                    fig.update_yaxes(ticks="")
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
        titulo_panel("Variables más importantes", f"Cuánto empeora el accuracy de {mejor} al desordenar cada variable")
        imp = pd.DataFrame({"Variable": [utils.ETIQUETAS[k] for k in m["importancias"]], "Importancia": list(m["importancias"].values())})
        fig = px.bar(imp.sort_values("Importancia"), x="Importancia", y="Variable", orientation="h", color_discrete_sequence=[ACENTO])
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.4, yaxis_title=None)
        fig.update_xaxes(showgrid=True, gridcolor="#E8EBEF")
        fig.update_yaxes(showgrid=False)
        mostrar(estilo(fig, 300))
    with c2, st.container(key="panel_proceso"):
        titulo_panel("Proceso seguido")
        st.markdown(
            """
1. **Dataset:** 500 clientes con 6 variables y el campo abandono (sí o no).
2. **Limpieza con Pandas:** duplicados, valores imposibles y nulos (mediana).
3. **X e y:** 6 variables de entrada y `abandono` como objetivo.
4. **Entrenamiento y prueba:** 80 % y 20 %, con estratificación.
5. **Modelos:** Regresión Logística, Random Forest y KNN.
6. **Evaluación:** accuracy, matriz de confusión, reporte y validación cruzada.
7. **Selección:** el mejor se guarda en `ml/modelo.pkl` y lo usa esta aplicación.
            """
        )


paginas = st.navigation([
    st.Page(pagina_dashboard, title="Dashboard", url_path="dashboard", default=True),
    st.Page(pagina_prediccion, title="Nueva predicción", url_path="prediccion"),
    st.Page(pagina_historial, title="Historial", url_path="historial"),
    st.Page(pagina_modelo, title="Análisis del modelo", url_path="modelo"),
])

_m = get_metricas()
st.sidebar.markdown(
    '<div class="side-card"><div class="t">Modelo activo</div>'
    f'<div class="f"><span>Algoritmo</span><span>{_m["modelo_seleccionado"]}</span></div>'
    f'<div class="f"><span>Accuracy</span><span>{_m["resultados"][_m["modelo_seleccionado"]]["accuracy"]:.1%}</span></div>'
    f'<div class="f"><span>Clientes en la base</span><span>{_m["n_total"]}</span></div></div>',
    unsafe_allow_html=True,
)

paginas.run()
