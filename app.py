"""Aplicación de pronóstico semanal para el proyecto UCI Online Retail."""

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "modelos" / "regresion_lineal.joblib"
RUTA_HISTORIAL = BASE / "data" / "historial_reciente.csv"
RUTA_PUNTOS = BASE / "data" / "puntos_grafica.csv"


@st.cache_resource
def cargar_modelo():
    """Carga una sola vez el modelo entrenado en el Paso 3."""
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_historial():
    """Carga los últimos cuatro valores observados de cada producto."""
    return pd.read_csv(RUTA_HISTORIAL, dtype={"StockCode": "string"})


@st.cache_data
def cargar_puntos():
    """Carga ejemplos usados para dibujar la nube de puntos."""
    return pd.read_csv(RUTA_PUNTOS)


def pronosticar(modelo_guardado, ventas):
    """Ordena las cuatro entradas igual que durante el entrenamiento."""
    entrada = pd.DataFrame([ventas], columns=modelo_guardado["variables"])
    prediccion = float(modelo_guardado["modelo"].predict(entrada)[0])
    return max(0.0, prediccion)


st.set_page_config(
    page_title="Pronóstico semanal de ventas",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Pronóstico semanal de ventas")
st.caption("Proyecto 3 · Analítica comercial · UCI Online Retail")
st.write(
    "Estima cuántas unidades se venderán **la semana siguiente** "
    "a partir de las ventas de las cuatro semanas anteriores."
)

try:
    modelo_guardado = cargar_modelo()
    historial = cargar_historial()
    puntos = cargar_puntos()
except (FileNotFoundError, OSError) as error:
    st.error(f"Falta un archivo necesario para iniciar la aplicación: {error}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Productos de ejemplo", f"{len(historial)}")
col2.metric("Semanas usadas", "4")
col3.metric("Error medio en prueba", "144,59 unidades")

etiquetas = {
    fila.StockCode: f"{fila.StockCode} · {fila.Description}"
    for fila in historial.itertuples(index=False)
}
codigos = historial["StockCode"].tolist()
indice_inicial = codigos.index("85099B") if "85099B" in codigos else 0
codigo = st.selectbox(
    "Elige un producto para cargar sus ventas históricas",
    options=codigos,
    index=indice_inicial,
    format_func=lambda valor: etiquetas[valor],
)
fila = historial.loc[historial["StockCode"] == codigo].iloc[0]
st.info(
    "El código del producto solo carga valores de ejemplo. "
    "El modelo calcula la predicción usando las **cuatro ventas semanales**, "
    "que puedes modificar abajo. La gráfica y el pronóstico cambian al ajustar los controles."
)
st.caption(f"Ejemplo histórico: últimas semanas completas hasta {fila['SemanaUltima']}.")

ventas = {}
etiquetas_semanas = {
    3: "Hace 4 semanas",
    2: "Hace 3 semanas",
    1: "Hace 2 semanas",
    0: "Última semana",
}
columnas = st.columns(4)
rango_slider = max(500, 2 * max(int(fila[f"ventas_t_{i}"]) for i in range(4)))
for indice, columna in zip((3, 2, 1, 0), columnas):
    variable = f"ventas_t_{indice}"
    with columna:
        if indice == 0:
            ventas[variable] = st.slider(
                etiquetas_semanas[indice],
                min_value=0,
                max_value=rango_slider,
                value=int(fila[variable]),
                step=1,
                key=f"{codigo}_{variable}",
            )
        else:
            ventas[variable] = st.number_input(
                etiquetas_semanas[indice],
                min_value=0,
                value=int(fila[variable]),
                step=1,
                key=f"{codigo}_{variable}",
            )

estimacion = pronosticar(modelo_guardado, ventas)
unidades = round(estimacion)
st.metric("Unidades estimadas para la semana siguiente", f"{unidades:,}".replace(",", "."))

st.subheader("Cómo responde la regresión lineal")
st.caption(
    "Azul: ejemplos históricos. Rojo: predicción del modelo al variar la última semana "
    "y mantener fijas las otras tres. Naranja: tu pronóstico actual. "
    "Mueve el control de la última semana para ver cómo cambia."
)

# Una regresión con cuatro entradas no cabe completa en un plano.
# Para dibujar una recta, variamos una entrada y fijamos las otras tres.
limite_x = max(1000, int(ventas["ventas_t_0"] * 1.2))
valores_x = np.linspace(0, limite_x, 120)
entradas_recta = pd.DataFrame(
    [{**ventas, "ventas_t_0": valor} for valor in valores_x],
    columns=modelo_guardado["variables"],
)
valores_y = np.clip(modelo_guardado["modelo"].predict(entradas_recta), 0, None)
recta = pd.DataFrame({"ventas_t_0": valores_x, "prediccion": valores_y})
limite_y = max(1000, int(max(float(valores_y.max()), estimacion) * 1.2))
puntos_visibles = puntos.loc[
    (puntos["ventas_t_0"] <= limite_x)
    & (puntos["unidades_siguiente_semana"] <= limite_y)
]

eje_x = alt.X("ventas_t_0:Q", title="Ventas de la última semana", scale=alt.Scale(domain=[0, limite_x]))
nube = alt.Chart(puntos_visibles).mark_circle(size=24, opacity=0.18, color="#60a5fa").encode(
    x=eje_x,
    y=alt.Y(
        "unidades_siguiente_semana:Q",
        title="Ventas de la semana siguiente",
        scale=alt.Scale(domain=[0, limite_y]),
    ),
)
linea = alt.Chart(recta).mark_line(color="#ef4444", strokeWidth=3).encode(
    x=eje_x,
    y=alt.Y("prediccion:Q", scale=alt.Scale(domain=[0, limite_y])),
)
punto_actual = alt.Chart(
    pd.DataFrame({"ventas_t_0": [ventas["ventas_t_0"]], "prediccion": [estimacion]})
).mark_circle(size=220, color="#f59e0b", stroke="white", strokeWidth=2).encode(
    x=eje_x,
    y=alt.Y("prediccion:Q", scale=alt.Scale(domain=[0, limite_y])),
    tooltip=[
        alt.Tooltip("ventas_t_0:Q", title="Última semana", format=".0f"),
        alt.Tooltip("prediccion:Q", title="Pronóstico", format=".1f"),
    ],
)
st.altair_chart((nube + linea + punto_actual).properties(height=380), width="stretch")

grafico = pd.DataFrame(
    {
        "Semana": [
            "Hace 4 semanas", "Hace 3 semanas", "Hace 2 semanas",
            "Última semana", "Semana siguiente",
        ],
        "Ventas registradas": [
            ventas["ventas_t_3"], ventas["ventas_t_2"],
            ventas["ventas_t_1"], ventas["ventas_t_0"], None,
        ],
        "Pronóstico": [None, None, None, None, unidades],
    }
)
st.subheader("Ventas recientes y pronóstico")
st.bar_chart(grafico.set_index("Semana"))
st.caption(
    "Es una estimación de ventas registradas, no una garantía. "
    "El modelo no conoce precios futuros, promociones ni faltantes de inventario."
)

with st.expander("¿Cómo se obtuvo este modelo?"):
    st.write(
        "Se entrenaron una regresión lineal y un Random Forest con 100 productos "
        "frecuentes. La regresión lineal obtuvo el menor error en las ocho semanas "
        "de prueba: 144,59 unidades de error absoluto medio por producto y semana."
    )
    st.write(
        "Las cuatro entradas son ventas registradas de semanas pasadas. "
        "La predicción corresponde solo a la siguiente semana."
    )

st.divider()
st.caption(
    "Fuente: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online-retail) "
    "(Chen, 2015; CC BY 4.0). Proyecto académico de Aprendizaje Automático."
)
