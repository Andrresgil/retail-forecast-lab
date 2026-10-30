# Pronóstico semanal de ventas con Streamlit

Esta aplicación usa la regresión lineal entrenada en `ACA1.ipynb` para estimar las unidades vendidas de un producto en la semana siguiente. Recibe las ventas de las cuatro semanas anteriores. El código del producto sirve para cargar un ejemplo histórico; no es una entrada del modelo.

## Archivos para el repositorio

```text
app.py
requirements.txt
data/historial_reciente.csv
modelos/regresion_lineal.joblib
README.md
```

La carpeta ya contiene todo lo necesario para ejecutar la aplicación. El CSV tiene solo 100 productos y sus cuatro últimas semanas completas; el archivo Excel original no se necesita en el despliegue.

## Probar en tu computador

Usa Python 3.12 desde esta carpeta:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente `http://localhost:8501`. Elige un producto, cambia las cuatro cifras si quieres y pulsa **Calcular pronóstico**.

## Publicar con GitHub y Streamlit Community Cloud

1. Crea un repositorio público nuevo en GitHub.
2. Sube **el contenido de esta carpeta** a la raíz del repositorio, conservando las carpetas `data` y `modelos`.
3. Entra a [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión con GitHub.
4. Selecciona **Create app**, el repositorio, la rama `main` y el archivo principal `app.py`.
5. En **Advanced settings**, elige Python **3.12**. Pulsa **Deploy**.
6. Cuando abra la app, copia su URL pública para la infografía y la entrega.

Si el despliegue falla, revisa primero los registros de la aplicación en Community Cloud. Confirma que `requirements.txt` está en la raíz, que `app.py` es el archivo principal y que las carpetas `data` y `modelos` se subieron completas.

## Alcance

El error absoluto medio en las ocho semanas de prueba fue de **144,59 unidades por producto y semana**. El modelo se entrenó con 100 productos frecuentes del conjunto UCI Online Retail. Predice ventas registradas para una semana, no demanda no observada ni un período más largo.

**Fuente de datos:** Chen, D. (2015). *Online Retail*. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licencia CC BY 4.0.
