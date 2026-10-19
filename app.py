# Bibliotecas.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Configuración de la página.
st.set_page_config(
    page_title="CrediScope",
    page_icon="📊",
    layout="wide"
)

BASE = Path(__file__).resolve().parent


# Lectura de resultados existentes.
@st.cache_data
def cargar_datos():
    datos = pd.read_csv(BASE / "predicciones_prueba.csv")

    with open(
        BASE / "parametros_economicos.json",
        encoding="utf-8"
    ) as archivo:
        parametros = json.load(archivo)

    curva_cv = pd.read_csv(BASE / "curva_costos_validacion.csv")

    columnas = {
        "Clase_real",
        "Probabilidad_incumplimiento"
    }

    if not columnas.issubset(datos.columns):
        raise ValueError(
            "El CSV de predicciones no contiene las columnas esperadas."
        )

    if not {"Umbral", "Costo_medio_CV_USD"}.issubset(curva_cv.columns):
        raise ValueError(
            "La curva debe ser la de validación: "
            "Umbral y Costo_medio_CV_USD."
        )

    y = pd.to_numeric(datos["Clase_real"], errors="raise")
    p = pd.to_numeric(
        datos["Probabilidad_incumplimiento"],
        errors="raise"
    )

    if len(datos) == 0:
        raise ValueError("El archivo de predicciones está vacío.")

    if not y.isin([0, 1]).all():
        raise ValueError("Las clases reales deben ser 0 o 1.")

    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Las probabilidades deben estar entre 0 y 1.")

    for clave in ["C_FP", "C_FN", "umbral_empirico"]:
        if clave not in parametros:
            raise ValueError(f"Falta el parámetro {clave}.")

    if (
        float(parametros["C_FP"]) <= 0
        or float(parametros["C_FN"]) <= 0
    ):
        raise ValueError("Los costos del escenario base deben ser positivos.")

    if not 0 <= float(parametros["umbral_empirico"]) <= 1:
        raise ValueError("El umbral empírico debe estar entre 0 y 1.")

    # Ordenar las probabilidades permite calcular los escenarios rápidamente.
    orden = np.argsort(p.to_numpy())
    probabilidades = p.to_numpy()[orden]
    clases = y.to_numpy(dtype=int)[orden]

    positivos_acumulados = np.concatenate(
        ([0], np.cumsum(clases))
    )

    return probabilidades, positivos_acumulados, parametros, curva_cv


try:
    probabilidades, acumulados, parametros, curva_cv = cargar_datos()
except (OSError, ValueError, KeyError) as error:
    st.error(f"No se pudieron cargar los archivos: {error}")
    st.info(
        "Comprueba que los tres archivos estén junto a app.py "
        "y que sus nombres coincidan exactamente."
    )
    st.stop()


# Conteos de clasificación para uno o varios umbrales.
def conteos(umbrales):
    posiciones = np.searchsorted(
        probabilidades,
        umbrales,
        side="left"
    )

    fn = acumulados[posiciones]
    tn = posiciones - fn

    total_positivos = acumulados[-1]
    total_negativos = len(probabilidades) - total_positivos

    tp = total_positivos - fn
    fp = total_negativos - tn

    return tn, fp, fn, tp


def evaluar(umbral, costo_fp, costo_fn):
    tn, fp, fn, tp = [
        int(valor) for valor in conteos(float(umbral))
    ]

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Recall": recall,
        "Precisión": precision,
        "F1": f1,
        "Costo total (USD)": fp * costo_fp + fn * costo_fn
    }


def numero(valor, decimales=0):
    texto = f"{valor:,.{decimales}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


# Valores del escenario base.
base_fp = float(parametros["C_FP"])
base_fn = float(parametros["C_FN"])
umbral_cv = float(parametros["umbral_empirico"])


# Presentación.
st.title("CrediScope")
st.subheader("Riesgo crediticio y decisiones de financiamiento")
st.write(
    "Explora cómo los costos de los errores y el umbral de decisión "
    "modifican los resultados de LightGBM sobre los préstamos de prueba."
)

st.caption(
    f"{numero(len(probabilidades))} préstamos de prueba · "
    "Clase 1: incumplimiento · Clase 0: préstamo pagado"
)


# Controles.
with st.sidebar:
    st.header("Configura el escenario")

    costo_fp = st.number_input(
        "Costo de falso positivo (USD)",
        min_value=1.0,
        value=base_fp,
        step=500.0,
        help="Beneficio perdido por clasificar como riesgoso un préstamo pagador."
    )

    costo_fn = st.number_input(
        "Costo de falso negativo (USD)",
        min_value=1.0,
        value=base_fn,
        step=500.0,
        help="Costo supuesto de clasificar como pagador un préstamo incumplido."
    )

    modo = st.radio(
        "Regla de decisión",
        [
            "Empírico del escenario base",
            "Teórico según costos",
            "Manual",
            "Estándar 0,5"
        ]
    )

    manual = st.slider(
        "Umbral manual",
        min_value=0.0,
        max_value=1.0,
        value=0.2856,
        step=0.0001,
        format="%.4f",
        disabled=modo != "Manual"
    )

    st.caption(
        "Una probabilidad igual o superior al umbral se clasifica como riesgo."
    )


teorico = costo_fp / (costo_fp + costo_fn)

if modo == "Empírico del escenario base":
    umbral = umbral_cv
elif modo == "Teórico según costos":
    umbral = teorico
elif modo == "Manual":
    umbral = manual
else:
    umbral = 0.5

es_base = (
    np.isclose(costo_fp, base_fp)
    and np.isclose(costo_fn, base_fn)
)

if not es_base:
    st.info(
        "Modificaste los costos. El umbral empírico guardado corresponde "
        "al escenario base; no se ha optimizado nuevamente para estos costos. "
        "El umbral teórico sí se actualiza."
    )

st.write(
    f"**Umbral aplicado:** {umbral:.6f} · "
    f"**Umbral teórico:** {teorico:.6f} · "
    f"**Relación costo FN/FP:** {costo_fn / costo_fp:.2f}"
)


# Indicadores.
resultado = evaluar(umbral, costo_fp, costo_fn)
referencia = evaluar(0.5, costo_fp, costo_fn)

costo = resultado["Costo total (USD)"]
costo_base_actual = referencia["Costo total (USD)"]
ahorro = costo_base_actual - costo
ahorro_porcentaje = (
    100 * ahorro / costo_base_actual
    if costo_base_actual else 0.0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("Recall", f"{numero(resultado['Recall'] * 100, 2)} %")
c2.metric("Precisión", f"{numero(resultado['Precisión'] * 100, 2)} %")
c3.metric("Costo total", f"{numero(costo)} USD")
c4.metric(
    "Ahorro frente a 0,5",
    f"{numero(ahorro)} USD",
    delta=f"{numero(ahorro_porcentaje, 2)} %"
)

st.caption(
    "El ahorro compara ambas reglas con los mismos costos seleccionados. "
    "Un ahorro negativo indica un costo superior al de usar 0,5."
)


# Matriz de confusión y costos.
izquierda, derecha = st.columns(2)

with izquierda:
    st.subheader("Matriz de confusión")

    matriz = np.array([
        [resultado["TN"], resultado["FP"]],
        [resultado["FN"], resultado["TP"]]
    ])

    etiquetas = [
        [
            f"TN\n{numero(resultado['TN'])}\n0 USD",
            f"FP\n{numero(resultado['FP'])}\n"
            f"{numero(resultado['FP'] * costo_fp)} USD"
        ],
        [
            f"FN\n{numero(resultado['FN'])}\n"
            f"{numero(resultado['FN'] * costo_fn)} USD",
            f"TP\n{numero(resultado['TP'])}\n0 USD"
        ]
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(matriz, cmap="Blues")

    for fila in range(2):
        for columna in range(2):
            color = (
                "white"
                if matriz[fila, columna] > matriz.max() / 2
                else "black"
            )
            ax.text(
                columna,
                fila,
                etiquetas[fila][columna],
                ha="center",
                va="center",
                color=color,
                fontsize=10
            )

    ax.set_xticks([0, 1], ["Pagador", "Riesgoso"])
    ax.set_yticks([0, 1], ["Pagado", "Incumplido"])
    ax.set_xlabel("Clasificación del modelo")
    ax.set_ylabel("Resultado real")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with derecha:
    st.subheader("Composición del costo")

    componentes = pd.DataFrame({
        "Tipo de error": ["Falsos positivos", "Falsos negativos"],
        "Cantidad": [resultado["FP"], resultado["FN"]],
        "Costo unitario (USD)": [costo_fp, costo_fn],
        "Costo total (USD)": [
            resultado["FP"] * costo_fp,
            resultado["FN"] * costo_fn
        ]
    })

    st.dataframe(componentes, hide_index=True)
    st.metric("F1", numero(resultado["F1"], 4))

    st.write(
        "Reducir el umbral permite detectar más incumplimientos, "
        "pero también puede aumentar los préstamos pagadores "
        "clasificados como riesgosos."
    )


# Curva exploratoria sobre prueba.
st.subheader("Costo en prueba según el umbral")

umbrales = np.unique(
    np.concatenate([
        np.linspace(0, 1, 501),
        [0.5, teorico, umbral_cv, umbral]
    ])
)

_, fp_curva, fn_curva, _ = conteos(umbrales)
costos = fp_curva * costo_fp + fn_curva * costo_fn

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(umbrales, costos, color="#2563eb", label="Costo en prueba")
ax.axvline(0.5, color="gray", linestyle="--", label="Estándar 0,5")
ax.axvline(
    teorico,
    color="darkorange",
    linestyle=":",
    label=f"Teórico: {teorico:.4f}"
)
ax.axvline(
    umbral_cv,
    color="seagreen",
    linestyle="--",
    label=f"Empírico base: {umbral_cv:.4f}"
)
ax.scatter(
    [umbral],
    [costo],
    color="crimson",
    zorder=5,
    label="Escenario aplicado"
)

ax.set_xlabel("Umbral")
ax.set_ylabel("Costo total (USD)")
ax.ticklabel_format(axis="y", style="plain")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Esta curva muestra resultados de prueba para explorar escenarios. "
    "No se utiliza su mínimo para seleccionar la regla final. "
    "El umbral final se eligió previamente mediante validación en entrenamiento."
)


# Curva que respalda la selección original.
with st.expander("Ver la selección del umbral en validación", expanded=False):
    st.write(
        f"Escenario base: FP = {numero(base_fp)} USD y "
        f"FN = {numero(base_fn)} USD. "
        "Esta curva permanece fija al cambiar los controles."
    )

    curva_cv = curva_cv.sort_values("Umbral")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        curva_cv["Umbral"],
        curva_cv["Costo_medio_CV_USD"],
        color="#2563eb"
    )
    ax.axvline(0.5, color="gray", linestyle="--", label="Estándar 0,5")
    ax.axvline(
        umbral_cv,
        color="seagreen",
        linestyle="--",
        label=f"Mínimo seleccionado en CV: {umbral_cv:.4f}"
    )

    ax.set_xlabel("Umbral")
    ax.set_ylabel("Costo medio por pliegue (USD)")
    ax.ticklabel_format(axis="y", style="plain")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "El costo medio por pliegue no es directamente comparable "
        "con el costo total de prueba."
    )


# Comparación de reglas con los mismos costos.
st.subheader("Comparación de reglas")

reglas = {
    "Estándar 0,5": 0.5,
    "Teórico según costos": teorico,
    "Empírico del escenario base": umbral_cv,
    "Escenario aplicado": umbral
}

filas = []

for nombre, corte in reglas.items():
    valores = evaluar(corte, costo_fp, costo_fn)
    filas.append({
        "Regla": nombre,
        "Umbral": corte,
        **valores,
        "Ahorro frente a 0,5 (USD)": (
            costo_base_actual - valores["Costo total (USD)"]
        )
    })

tabla = pd.DataFrame(filas)
st.dataframe(tabla.round(4), hide_index=True)

st.download_button(
    "Descargar comparación del escenario",
    data=tabla.to_csv(index=False).encode("utf-8-sig"),
    file_name="crediscope_comparacion.csv",
    mime="text/csv"
)


# Contexto y límites.
with st.expander("Cómo interpretar los resultados"):
    st.markdown(
        """
        **Recall:** proporción de incumplimientos detectados.

        **Precisión:** proporción de préstamos realmente incumplidos
        entre los clasificados como riesgosos.

        **Falso positivo:** préstamo pagador clasificado como riesgoso.

        **Falso negativo:** préstamo incumplido clasificado como pagador.

        **Costo total:** FP × costo FP + FN × costo FN.

        Los costos representan supuestos del escenario académico.
        Los ahorros no son beneficios realizados por una institución.

        El umbral teórico supone probabilidades calibradas, costos constantes
        y costo cero para las clasificaciones correctas. No se verificó
        formalmente la calibración.

        El análisis utiliza préstamos históricos aprobados. No garantiza
        resultados equivalentes para nuevas solicitudes o períodos.""")

st.caption(
    "CrediScope · Certamen 1 de Machine Learning · "
    "Hellen Quiñones y Gustavo Sánchez")
