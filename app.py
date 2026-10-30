"""Aplicación de pronóstico semanal para el proyecto UCI Online Retail."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "modelos" / "regresion_lineal.joblib"
RUTA_HISTORIAL = BASE / "data" / "historial_reciente.csv"


@st.cache_resource
def cargar_modelo():
    """Carga una sola vez el modelo entrenado en el Paso 3."""
    return joblib.load(RUTA_MODELO)


@st.cache_data
def cargar_historial():
    """Carga los últimos cuatro valores observados de cada producto."""
    return pd.read_csv(RUTA_HISTORIAL, dtype={"StockCode": "string"})


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
    "que puedes modificar abajo."
)
st.caption(f"Ejemplo histórico: últimas semanas completas hasta {fila['SemanaUltima']}.")

ventas = {}
etiquetas_semanas = {
    3: "Hace 4 semanas",
    2: "Hace 3 semanas",
    1: "Hace 2 semanas",
    0: "Última semana",
}
with st.form(key=f"pronostico_{codigo}"):
    columnas = st.columns(4)
    for indice, columna in zip((3, 2, 1, 0), columnas):
        variable = f"ventas_t_{indice}"
        with columna:
            ventas[variable] = st.number_input(
                etiquetas_semanas[indice],
                min_value=0,
                value=int(fila[variable]),
                step=1,
                key=f"{codigo}_{variable}",
            )
    calcular = st.form_submit_button("Calcular pronóstico", type="primary")

if calcular:
    estimacion = pronosticar(modelo_guardado, ventas)
    unidades = round(estimacion)
    st.metric("Unidades estimadas para la semana siguiente", f"{unidades:,}".replace(",", "."))

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
