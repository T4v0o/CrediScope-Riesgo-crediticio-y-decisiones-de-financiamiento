# Bibliotecas.
import json
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st


# Configuración de la página.
st.set_page_config(page_title="CrediScope", page_icon="📊", layout="wide")
BASE = Path(__file__).resolve().parent
MODOS = [
    "Empírico del escenario base", "Teórico según costos", "Manual", "Estándar 0,5"
]


# Lectura y validación de los resultados exportados desde el notebook.
@st.cache_data
def cargar_datos():
    datos = pd.read_csv(BASE / "predicciones_prueba.csv")
    with open(BASE / "parametros_economicos.json", encoding="utf-8") as archivo:
        parametros = json.load(archivo)
    curva_cv = pd.read_csv(BASE / "curva_costos_validacion.csv")

    if not {"Clase_real", "Probabilidad_incumplimiento"}.issubset(datos.columns):
        raise ValueError("Faltan Clase_real o Probabilidad_incumplimiento en el CSV.")
    if datos.empty:
        raise ValueError("El archivo de predicciones está vacío.")
    y = pd.to_numeric(datos["Clase_real"], errors="raise")
    p = pd.to_numeric(datos["Probabilidad_incumplimiento"], errors="raise")
    if not y.isin([0, 1]).all():
        raise ValueError("Las clases reales deben ser 0 o 1.")
    if not np.isfinite(p).all() or not p.between(0, 1).all():
        raise ValueError("Las probabilidades deben ser finitas y estar entre 0 y 1.")

    for clave in ["C_FP", "C_FN", "umbral_empirico"]:
        if clave not in parametros:
            raise ValueError(f"Falta el parámetro {clave}.")
        parametros[clave] = float(parametros[clave])
        if not np.isfinite(parametros[clave]):
            raise ValueError(f"El parámetro {clave} debe ser finito.")
    if min(parametros["C_FP"], parametros["C_FN"]) < 1:
        raise ValueError("Los costos base deben ser de al menos 1 USD.")
    if not 0 <= parametros["umbral_empirico"] <= 1:
        raise ValueError("El umbral empírico debe estar entre 0 y 1.")

    columnas_cv = ["Umbral", "Costo_medio_CV_USD"]
    if not set(columnas_cv).issubset(curva_cv.columns) or curva_cv.empty:
        raise ValueError("La curva de validación requiere Umbral y Costo_medio_CV_USD.")
    curva_cv = curva_cv[columnas_cv].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(curva_cv.to_numpy()).all():
        raise ValueError("La curva de validación contiene valores no finitos.")
    if not curva_cv["Umbral"].between(0, 1).all():
        raise ValueError("Los umbrales de validación deben estar entre 0 y 1.")
    if (curva_cv["Costo_medio_CV_USD"] < 0).any():
        raise ValueError("Los costos de validación deben ser positivos o cero.")

    # Búsqueda ordenada: permite explorar umbrales sin volver a entrenar.
    orden = np.argsort(p.to_numpy())
    probabilidades = p.to_numpy()[orden]
    clases = y.to_numpy(dtype=int)[orden]
    acumulados = np.concatenate(([0], np.cumsum(clases)))
    return probabilidades, acumulados, parametros, curva_cv.sort_values("Umbral")


def conteos(umbrales):
    # La igualdad pertenece a la clase de riesgo: probabilidad >= umbral.
    posiciones = np.searchsorted(probabilidades, umbrales, side="left")
    fn = acumulados[posiciones]
    tn = posiciones - fn
    tp = acumulados[-1] - fn
    fp = len(probabilidades) - acumulados[-1] - tn
    return tn, fp, fn, tp


def evaluar(umbral, costo_fp, costo_fn):
    tn, fp, fn, tp = [int(valor) for valor in conteos(float(umbral))]
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Recall": tp / (tp + fn) if tp + fn else 0.0,
        "Precisión": tp / (tp + fp) if tp + fp else 0.0,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "Costo total (USD)": fp * costo_fp + fn * costo_fn,
    }


def numero(valor, decimales=0):
    return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def porcentaje(valor):
    return f"{numero(valor * 100, 2)} %"


def sincronizar(origen, destino):
    st.session_state[destino] = float(st.session_state[origen])


def restaurar_base():
    for prefijo, valor in [("fp", base_fp), ("fn", base_fn)]:
        st.session_state[f"{prefijo}_slider"] = valor
        st.session_state[f"{prefijo}_exacto"] = valor
    st.session_state["modo"] = MODOS[0]
    st.session_state["manual"] = float(round(umbral_cv, 4))


def control_costo(prefijo, titulo, base, ayuda):
    # Ambos controles comparten valor mediante callbacks, con claves distintas.
    maximo = float(max(200_000, np.ceil(base / 10_000) * 10_000))
    for sufijo in ["slider", "exacto"]:
        st.session_state.setdefault(f"{prefijo}_{sufijo}", base)
    st.slider(
        titulo, min_value=1.0, max_value=maximo, step=1.0, format="%.2f",
        key=f"{prefijo}_slider", help=ayuda,
        on_change=sincronizar, args=(f"{prefijo}_slider", f"{prefijo}_exacto"),
    )
    valor = st.number_input(
        f"Monto exacto de {prefijo.upper()} (USD)",
        min_value=1.0, max_value=maximo, step=10.0, format="%.2f",
        key=f"{prefijo}_exacto", on_change=sincronizar,
        args=(f"{prefijo}_exacto", f"{prefijo}_slider"),
    )
    st.caption(ayuda)
    return valor


def tarjeta(titulo, valor, descripcion, color="#2563EB"):
    # HTML propio con ajuste de línea para que no se corten los montos.
    st.markdown(
        f'<div class="cs-card" style="border-top-color:{color}">'
        f'<div class="cs-label">{escape(titulo)}</div>'
        f'<div class="cs-value">{escape(valor)}</div>'
        f'<div class="cs-detail">{escape(descripcion)}</div></div>',
        unsafe_allow_html=True,
    )


def estilo_ejes(ax, etiqueta_y):
    ax.set_facecolor("#FFFFFF")
    ax.set_xlabel("Umbral de clasificación")
    ax.set_ylabel(etiqueta_y)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: numero(x, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: numero(y, 1)))
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#CBD5E1")


# Carga de datos y valores base.
try:
    probabilidades, acumulados, parametros, curva_cv = cargar_datos()
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"No se pudieron cargar los resultados: {error}")
    st.info("Revisa los tres archivos de datos y sus nombres en la carpeta de app.py.")
    st.stop()

base_fp = float(parametros["C_FP"])
base_fn = float(parametros["C_FN"])
umbral_cv = float(parametros["umbral_empirico"])


# Estilo visual. El tema general se define en .streamlit/config.toml.
st.markdown("""
<style>
.cs-hero {background:linear-gradient(115deg,#123B68,#176B83); padding:1.7rem 2rem;
border-radius:18px; margin-bottom:1rem; color:white;}
.cs-hero h1 {color:white; margin:0; padding:0; font-size:2.8rem;}
.cs-hero p {color:#E6F2FF; margin:.6rem 0 0; font-size:1.05rem;}
.cs-card {background:#F8FAFC; color:#172B4D; border:1px solid #DFE7EF;
border-top:4px solid; border-radius:14px; padding:1.1rem 1.25rem; margin:.3rem 0 .8rem;}
.cs-label {font-size:1rem; font-weight:600;}
.cs-value {font-size:clamp(1.5rem,2.5vw,2.25rem); font-weight:700;
line-height:1.3; margin:.5rem 0; overflow-wrap:anywhere;}
.cs-detail {font-size:.9rem; color:#475569; line-height:1.5;}
</style>
<div class="cs-hero"><h1>CrediScope</h1>
<p>Riesgo crediticio y decisiones de financiamiento</p></div>
""", unsafe_allow_html=True)

st.write(
    "Explora el equilibrio entre detectar incumplimientos y evitar falsas alarmas. "
    "Compara cómo cambia el costo de los errores al modificar la regla de decisión."
)
st.caption(
    f"Modelo: LightGBM · {numero(len(probabilidades))} préstamos de prueba · "
    "Clase 1: incumplimiento · Clase 0: préstamo pagado"
)

with st.expander("Cómo usar el dashboard y qué significa cada regla"):
    st.markdown("""
1. Ajusta los costos en el panel izquierdo. Puedes mover los sliders o escribir el monto exacto.
2. Elige una regla. Si seleccionas **Manual**, podrás mover el slider del umbral.
3. Compara los errores y el costo con la referencia de **0,5**, usando los mismos costos.

**Umbral:** probabilidad mínima estimada de incumplimiento para clasificar un préstamo como riesgoso.

**Estándar 0,5:** marca como riesgosos los préstamos con probabilidad igual o superior al 50 %.

**Teórico según costos:** calcula el umbral como costo FP dividido por la suma de los costos FP y FN.
Se actualiza al cambiar los costos. Su justificación supone probabilidades calibradas.

**Empírico del escenario base:** aplica el umbral seleccionado mediante validación cruzada
con los costos originales. No vuelve a optimizarse al mover los controles.

**Manual:** permite explorar un umbral elegido por el usuario. Explorar prueba no constituye
una nueva selección validada del umbral.
""")


# Controles interactivos.
with st.sidebar:
    st.header("Configura el escenario")
    st.caption("Todos los costos son por préstamo y están expresados en USD.")
    st.button("Restaurar escenario base", on_click=restaurar_base, use_container_width=True)
    costo_fp = control_costo(
        "fp", "Costo de falso positivo (USD)", base_fp,
        "FP: préstamo pagado clasificado como riesgoso. Representa una oportunidad de beneficio perdida.",)
    costo_fn = control_costo(
        "fn", "Costo de falso negativo (USD)", base_fn,
        "FN: préstamo incumplido clasificado como pagador. Representa el costo supuesto del incumplimiento.",)
    st.divider()
    modo = st.radio("Regla de decisión", MODOS, key="modo")
    st.session_state.setdefault("manual", float(round(umbral_cv, 4)))
    manual = st.slider(
        "Umbral manual", min_value=0.0, max_value=1.0, step=0.0001,
        format="%.4f", key="manual", disabled=modo != "Manual",)
    if modo != "Manual":
        st.caption("El slider del umbral se activa al elegir Manual. Su valor no se aplica en las otras reglas.")
    st.caption(f"Costos base: FP {numero(base_fp, 2)} USD · FN {numero(base_fn, 2)} USD.")


# Regla aplicada y evaluación. Se preserva el umbral empírico con toda su precisión.
teorico = costo_fp / (costo_fp + costo_fn)
umbrales_por_modo = {
    MODOS[0]: umbral_cv, MODOS[1]: teorico, MODOS[2]: manual, MODOS[3]: 0.5}

umbral = umbrales_por_modo[modo]
es_base = np.isclose(costo_fp, base_fp) and np.isclose(costo_fn, base_fn)
resultado = evaluar(umbral, costo_fp, costo_fn)
referencia = evaluar(0.5, costo_fp, costo_fn)
costo = resultado["Costo total (USD)"]
costo_base_actual = referencia["Costo total (USD)"]
ahorro = costo_base_actual - costo
ahorro_porcentaje = 100 * ahorro / costo_base_actual if costo_base_actual else None

st.subheader("1. Resultado del escenario")
st.markdown(
    f"**Regla activa: {modo}.** Se clasifica como riesgoso un préstamo desde "
    f"una probabilidad estimada de incumplimiento de **{porcentaje(umbral)}**.")
st.caption(
    f"Umbral aplicado: {numero(umbral, 6)} · Teórico: {numero(teorico, 6)} · "
    f"Relación de costos FN/FP: {numero(costo_fn / costo_fp, 2)}")
if not es_base:
    st.info(
        "Estás explorando costos distintos de los originales. El umbral teórico y los costos "
        "se actualizan; el umbral empírico sigue siendo el seleccionado para el escenario base.")

c1, c2 = st.columns(2)
with c1:
    tarjeta("Incumplimientos detectados (Recall)", porcentaje(resultado["Recall"]),
            "Proporción de los incumplimientos reales que logra detectar la regla.")
with c2:
    tarjeta("Precisión de las alertas de riesgo", porcentaje(resultado["Precisión"]),
            "Proporción de incumplimientos reales entre los préstamos clasificados como riesgosos.", "#087F8C")
c3, c4 = st.columns(2)
with c3:
    tarjeta("Costo total del escenario", f"{numero(costo, 2)} USD",
            "Suma del costo de falsos positivos y falsos negativos.", "#D97706")
if ahorro_porcentaje is None:
    detalle_ahorro = "La referencia tiene costo cero; la variación porcentual no está definida."
elif ahorro > 0:
    detalle_ahorro = f"Reducción del costo: {numero(ahorro_porcentaje, 2)} %."
elif ahorro < 0:
    detalle_ahorro = f"Aumento del costo: {numero(abs(ahorro_porcentaje), 2)} %. El ahorro es negativo."
else:
    detalle_ahorro = "El costo es igual al de usar el umbral 0,5."
with c4:
    tarjeta("Ahorro frente al umbral 0,5", f"{numero(ahorro, 2)} USD",
            detalle_ahorro, "#16866A" if ahorro >= 0 else "#BE3455")
st.caption(
    f"La referencia de 0,5 cuesta {numero(costo_base_actual, 2)} USD con estos mismos costos. "
    "Los ahorros son estimaciones del escenario académico, no beneficios realizados.")

if resultado["TP"] + resultado["FN"] == 0:
    st.info("No hay incumplimientos reales en los datos; el recall no está definido y se muestra como 0 por convención.")
elif resultado["TP"] + resultado["FP"] == 0:
    st.info("Esta regla no marca ningún préstamo como riesgoso. La precisión no está definida y se muestra como 0 por convención.")
else:
    st.write(
        f"**Lectura del escenario:** se detectan {numero(resultado['TP'])} de "
        f"{numero(resultado['TP'] + resultado['FN'])} incumplimientos reales "
        f"({porcentaje(resultado['Recall'])}). Entre los préstamos clasificados como riesgosos, "
        f"el {porcentaje(resultado['Precisión'])} realmente incumplió.")


# Matriz con colores por tipo de resultado y costos por celda.
st.subheader("2. Errores de clasificación y sus costos")
izquierda, derecha = st.columns(2)
with izquierda:
    st.markdown("**Matriz de confusión**")
    etiquetas = [
        [f"Pagador identificado (TN)\n{numero(resultado['TN'])} préstamos\n0 USD",
         f"Falsa alarma (FP)\n{numero(resultado['FP'])} préstamos\n{numero(resultado['FP'] * costo_fp, 2)} USD"],
        [f"Incumplimiento\nno detectado (FN)\n{numero(resultado['FN'])} préstamos\n{numero(resultado['FN'] * costo_fn, 2)} USD",
         f"Incumplimiento\ndetectado (TP)\n{numero(resultado['TP'])} préstamos\n0 USD"],]
    
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.imshow([[0, 1], [2, 0]], cmap=ListedColormap(["#DBF1EA", "#FFF0CE", "#FCE0E4"]), vmin=0, vmax=2)
    for fila in range(2):
        for columna in range(2):
            ax.text(columna, fila, etiquetas[fila][columna], ha="center", va="center", color="#172B4D", fontsize=10)
    ax.set_xticks([0, 1], ["Pagador", "Riesgoso"])
    ax.set_yticks([0, 1], ["Pagado", "Incumplido"])
    ax.set_xlabel("Clasificación del modelo")
    ax.set_ylabel("Resultado real")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Verde: aciertos. Ámbar: falsos positivos. Rosado: falsos negativos. Los colores identifican tipos de resultado, no cantidades.")

with derecha:
    st.markdown("**Composición del costo**")
    componentes = pd.DataFrame({
        "Error": ["Falso positivo", "Falso negativo"],
        "Casos": [resultado["FP"], resultado["FN"]],
        "USD por caso": [costo_fp, costo_fn],
        "Costo (USD)": [resultado["FP"] * costo_fp, resultado["FN"] * costo_fn],
    })
    st.dataframe(componentes.style.format({
        "Casos": numero, "USD por caso": lambda x: numero(x, 2), "Costo (USD)": lambda x: numero(x, 2)
    }), hide_index=True, use_container_width=True)
    st.markdown("**Costo total = FP × costo FP + FN × costo FN**")
    st.metric("F1: equilibrio entre recall y precisión", numero(resultado["F1"], 4))
    st.caption("F1 varía entre 0 y 1. Resume recall y precisión, pero no incorpora los costos monetarios.")
    st.write(
        "Bajar el umbral marca más préstamos como riesgosos: puede detectar más incumplimientos "
        "y generar más falsas alarmas. Subirlo produce el efecto contrario. La precisión no tiene "
        "por qué cambiar de forma monótona."
    )


# Curva exploratoria de prueba. No se selecciona su mínimo.
st.subheader("3. Cómo cambia el costo al mover el umbral")
st.write("La curva muestra el costo calculado para cada umbral con los costos elegidos. El punto marca la regla activa.")
umbrales = np.unique(np.concatenate([np.linspace(0, 1, 501), [0.5, teorico, umbral_cv, umbral]]))
_, fp_curva, fn_curva, _ = conteos(umbrales)
costos = fp_curva * costo_fp + fn_curva * costo_fn
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(umbrales, costos / 1_000_000, color="#2563EB", linewidth=2.2, label="Costo en prueba")
ax.axvline(0.5, color="#64748B", linestyle="--", label="Estándar 0,5")
ax.axvline(teorico, color="#D97706", linestyle=":", linewidth=2, label=f"Teórico: {numero(teorico, 4)}")
ax.axvline(umbral_cv, color="#16866A", linestyle="--", label=f"Empírico base: {numero(umbral_cv, 4)}")
ax.scatter([umbral], [costo / 1_000_000], color="#BE3455", s=65, zorder=5, label="Regla activa")
estilo_ejes(ax, "Costo total (millones de USD)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=9)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)
st.caption(
    "La curva de prueba sirve para explorar escenarios. Su mínimo no se utiliza para seleccionar "
    "la regla final del trabajo. El umbral empírico fue seleccionado previamente mediante validación en entrenamiento."
)

with st.expander("Respaldo metodológico: selección del umbral en validación"):
    st.write(
        f"Costos originales: FP = {numero(base_fp, 2)} USD y FN = {numero(base_fn, 2)} USD. "
        f"Umbral seleccionado: {numero(umbral_cv, 6)}. Esta curva permanece fija al cambiar los controles."
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curva_cv["Umbral"], curva_cv["Costo_medio_CV_USD"] / 1_000_000, color="#2563EB", linewidth=2)
    ax.axvline(0.5, color="#64748B", linestyle="--", label="Estándar 0,5")
    ax.axvline(umbral_cv, color="#16866A", linestyle="--", label=f"Seleccionado en CV: {numero(umbral_cv, 4)}")
    estilo_ejes(ax, "Costo medio por pliegue (millones de USD)")
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Es el promedio del costo total de los pliegues, no un costo por préstamo. No es directamente comparable con el costo total de prueba.")


# Comparación sin duplicar la regla activa. El modo manual se añade solo al seleccionarlo.
st.subheader("4. Comparación de reglas")
reglas = {MODOS[3]: 0.5, MODOS[1]: teorico, MODOS[0]: umbral_cv}
if modo == "Manual":
    reglas["Manual"] = umbral
filas = []
for nombre, corte in reglas.items():
    valores = evaluar(corte, costo_fp, costo_fn)
    filas.append({
        "Regla": nombre, "Selección": "Actual" if nombre == modo else "",
        "Umbral": corte, **valores,
        "Ahorro frente a 0,5 (USD)": costo_base_actual - valores["Costo total (USD)"],
    })
tabla = pd.DataFrame(filas)
columnas_resumen = ["Regla", "Selección", "Umbral", "Recall", "Precisión", "Costo total (USD)", "Ahorro frente a 0,5 (USD)"]
formatos = {
    "Umbral": lambda x: numero(x, 4), "Recall": porcentaje, "Precisión": porcentaje,
    "Costo total (USD)": lambda x: numero(x, 2), "Ahorro frente a 0,5 (USD)": lambda x: numero(x, 2),
}
st.dataframe(tabla[columnas_resumen].style.format(formatos), hide_index=True, use_container_width=True)
st.caption("La columna Selección identifica la regla aplicada. Todas las filas se evalúan sobre los mismos préstamos y con los mismos costos.")
with st.expander("Ver conteos y F1 de cada regla"):
    st.dataframe(tabla[["Regla", "TN", "FP", "FN", "TP", "F1"]].style.format({
        "TN": numero, "FP": numero, "FN": numero, "TP": numero, "F1": lambda x: numero(x, 4)
    }), hide_index=True, use_container_width=True)
    st.caption("TN: pagadores identificados. FP: falsas alarmas. FN: incumplimientos no detectados. TP: incumplimientos detectados.")
exportacion = tabla.assign(Costo_FP_USD=costo_fp, Costo_FN_USD=costo_fn)
st.download_button(
    "Descargar comparación del escenario (CSV)",
    data=exportacion.to_csv(index=False).encode("utf-8-sig"),
    file_name="crediscope_comparacion.csv", mime="text/csv",
)


# Alcance y límites de la interpretación.
with st.expander("5. Supuestos y límites del análisis"):
    st.markdown("""
**Costos del escenario.** Los costos FP y FN son supuestos académicos constantes por préstamo.
Las clasificaciones correctas tienen costo cero en esta matriz. Los ahorros no son beneficios
realizados por una institución.

**Umbral teórico.** Supone probabilidades calibradas y la estructura de costos indicada.
No se verificó formalmente la calibración; no se garantiza que minimice el costo observado en prueba.

**Validación y prueba.** El umbral empírico se seleccionó mediante validación cruzada sobre
entrenamiento para los costos base. No se optimiza nuevamente al cambiar los controles.
La exploración manual del conjunto de prueba no sustituye una validación independiente.

**Población estudiada.** Los datos corresponden a préstamos históricos aprobados. No se garantiza
el mismo desempeño para nuevas solicitudes, préstamos rechazados u otros períodos.

**Uso de la aplicación.** Se reutilizan predicciones ya exportadas del modelo LightGBM.
La app explora decisiones sobre esos préstamos; no evalúa solicitudes nuevas ni reemplaza
la revisión humana de una decisión de financiamiento.
""")
st.caption("CrediScope · Certamen 1 de Machine Learning · Hellen Quiñones y Gustavo Sánchez")
