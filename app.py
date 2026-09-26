"""
Random Forest — Clasificación de tensión de liquidez (capital de trabajo)
Aplicación Streamlit.   Ejecutar:  streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import date, datetime

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from src import datos as D
from src import graficas as G
from src import modelo as M

st.set_page_config(
    page_title="Random Forest · Tensión de liquidez",
    page_icon="💧",
    layout="wide",
)

PCT_ENTRENAMIENTO = 0.80

# Etiquetas legibles para el formulario
ETIQUETAS = {
    "Ventas_12M_MXN": "Ventas últimos 12 meses (MXN)",
    "Crecimiento_Ventas_pct": "Crecimiento de ventas (fracción)",
    "Margen_EBITDA_pct": "Margen EBITDA (fracción)",
    "DSO_dias": "DSO — días de cobranza",
    "DIO_dias": "DIO — días de inventario",
    "DPO_dias": "DPO — días de pago a proveedores",
    "CxC_MXN": "Cuentas por cobrar (MXN)",
    "Inventarios_MXN": "Inventarios (MXN)",
    "CxP_MXN": "Cuentas por pagar (MXN)",
    "Deuda_CP_MXN": "Deuda de corto plazo (MXN)",
    "Caja_MXN": "Caja (MXN)",
    "Linea_Credito_Disponible_MXN": "Línea de crédito disponible (MXN)",
    "Concentracion_Top5_Clientes_pct": "Concentración top-5 clientes (fracción)",
    "Morosidad_CxC_pct": "Morosidad de CxC (fracción)",
    "Inventario_Obsoleto_pct": "Inventario obsoleto (fracción)",
    "Volatilidad_Ventas_pct": "Volatilidad de ventas (fracción)",
}


# ---------------------------------------------------------------------------
# Funciones con caché
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar(archivo_bytes: bytes | None, nombre: str | None) -> pd.DataFrame:
    if archivo_bytes is None:
        return D.cargar_base()
    buf = io.BytesIO(archivo_bytes)
    buf.name = nombre
    return D.leer_archivo(buf)


@st.cache_resource(show_spinner="Entrenando Random Forest…")
def entrenar_modelo(huella: str, params: tuple, _X_train, _y_train):
    return M.entrenar(_X_train, _y_train, dict(params))


@st.cache_data(show_spinner="Calculando importancia por permutación…")
def perm_cache(clave: str, _pipe, _X_test, _y_test):
    return M.importancia_permutacion(_pipe, _X_test, _y_test, n_repeats=10)


@st.cache_resource(show_spinner="Calculando valores SHAP…")
def shap_cache(clave: str, _pipe, _X):
    return M.explicacion_shap(_pipe, _X)


def a_excel(df: pd.DataFrame, hoja: str = "Resultados") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name=hoja)
    return buf.getvalue()


def formato_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
st.sidebar.title("Configuración")

st.sidebar.subheader("1 · Datos")
fuente = st.sidebar.radio(
    "Base de entrenamiento",
    ["Base prototipo del repositorio", "Subir archivo (Excel/CSV)"],
    label_visibility="collapsed",
)
archivo = None
if fuente.startswith("Subir"):
    archivo = st.sidebar.file_uploader(
        "Archivo con la misma estructura que la base prototipo",
        type=["xlsx", "xls", "csv"],
        help="Si es Excel, se lee la hoja 'Datos_Modelo' (o la primera hoja si no existe).",
    )
    if archivo is None:
        st.sidebar.info("Mientras no suba un archivo se usa la base prototipo.")

st.sidebar.subheader("2 · Hiperparámetros")
if "params" not in st.session_state:
    st.session_state.params = dict(M.PARAMETROS_DEFAULT)

with st.sidebar.form("form_hiper"):
    p = st.session_state.params
    n_estimators = st.slider("n_estimators (número de árboles)", 50, 1000, p["n_estimators"], step=50)
    sin_limite = st.checkbox("max_depth sin límite", value=p["max_depth"] is None)
    max_depth = st.slider("max_depth (profundidad máxima)", 2, 30, p["max_depth"] or 10,
                          disabled=False, help="Se ignora si 'sin límite' está marcado.")
    min_samples_split = st.slider("min_samples_split", 2, 20, p["min_samples_split"])
    min_samples_leaf = st.slider("min_samples_leaf", 1, 20, p["min_samples_leaf"])
    opciones_mf = {"sqrt": "sqrt", "log2": "log2", "0.5 (50% de variables)": 0.5, "Todas (None)": None}
    etiqueta_mf = next(k for k, v in opciones_mf.items() if v == p["max_features"])
    max_features = st.selectbox("max_features", list(opciones_mf), index=list(opciones_mf).index(etiqueta_mf))
    criterion = st.selectbox("criterion", ["gini", "entropy"], index=["gini", "entropy"].index(p["criterion"]))
    bootstrap = st.toggle("bootstrap", value=p["bootstrap"])
    entrenar_btn = st.form_submit_button("Entrenar modelo", type="primary", width="stretch")
    reset_btn = st.form_submit_button("Restablecer valores default", width="stretch")

if entrenar_btn:
    st.session_state.params = {
        "n_estimators": n_estimators,
        "max_depth": None if sin_limite else max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": opciones_mf[max_features],
        "criterion": criterion,
        "bootstrap": bootstrap,
    }
if reset_btn:
    st.session_state.params = dict(M.PARAMETROS_DEFAULT)
    st.rerun()

st.sidebar.subheader("3 · Umbral de decisión")
umbral = st.sidebar.slider(
    "Probabilidad mínima para clasificar como 'con tensión'",
    0.05, 0.95, 0.50, 0.01,
    help="Bajar el umbral aumenta el recall (se detectan más casos) a costa de más falsas alarmas.",
)
st.sidebar.caption(
    f"class_weight='balanced' · random_state={M.RANDOM_STATE} · "
    f"partición temporal {PCT_ENTRENAMIENTO:.0%}/{1 - PCT_ENTRENAMIENTO:.0%}"
)

# ---------------------------------------------------------------------------
# Carga y preparación
# ---------------------------------------------------------------------------
try:
    df_raw = cargar(archivo.getvalue() if archivo else None, archivo.name if archivo else None)
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo leer el archivo: {e}")
    st.stop()

faltantes = D.validar_columnas(df_raw, D.REQUERIDAS_ENTRENAMIENTO)
if faltantes:
    st.error("Al archivo le faltan columnas requeridas: " + ", ".join(faltantes))
    st.stop()

df = D.preparar_base(df_raw)
if len(df) < 50 or df[D.TARGET].nunique() < 2:
    st.error("La base necesita al menos 50 observaciones válidas y ambas clases en el target.")
    st.stop()

X_train, X_test, y_train, y_test, df_train, df_test = D.particion_temporal(df, PCT_ENTRENAMIENTO)
if y_test.nunique() < 2:
    st.warning("El conjunto de prueba solo contiene una clase; algunas métricas no se podrán calcular.")

params = st.session_state.params
huella = str(pd.util.hash_pandas_object(df[D.PREDICTORES + [D.TARGET]], index=False).sum())
params_tuple = tuple(sorted(params.items(), key=lambda kv: kv[0]))
clave = huella + str(params_tuple)

pipe = entrenar_modelo(huella, params_tuple, X_train, y_train)
p_train = M.probabilidades(pipe, X_train)
p_test = M.probabilidades(pipe, X_test)

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.title("Random Forest · Predicción de tensión de liquidez")
st.caption(
    "Clasificación binaria de `Tension_Liquidez_bin` con indicadores de capital de trabajo. "
    "Base didáctica con datos simulados."
)

m_test = M.metricas(y_test, p_test, umbral)
k = st.columns(5)
for col, nombre in zip(k, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]):
    col.metric(f"{nombre} (prueba)", formato_pct(m_test[nombre]) if nombre != "ROC-AUC" else f"{m_test[nombre]:.3f}")

tabs = st.tabs([
    "Datos y partición",
    "Evaluación",
    "Importancia de variables",
    "SHAP",
    "Predicción individual",
    "Predicción masiva",
    "Exportar modelo",
])

# ---------------------------------------------------------------------------
# 1. Datos y partición
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Partición temporal")
    resumen = pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Prueba"],
        "Observaciones": [len(df_train), len(df_test)],
        "Desde": [df_train[D.COL_FECHA].min().date(), df_test[D.COL_FECHA].min().date()],
        "Hasta": [df_train[D.COL_FECHA].max().date(), df_test[D.COL_FECHA].max().date()],
        "% con tensión": [formato_pct(y_train.mean()), formato_pct(y_test.mean())],
    })
    st.dataframe(resumen, hide_index=True, width="stretch")
    st.caption(
        f"Los datos se ordenan por fecha: el {PCT_ENTRENAMIENTO:.0%} más antiguo entrena y el "
        f"{1 - PCT_ENTRENAMIENTO:.0%} más reciente se reserva como prueba, lo que simula el uso real del modelo."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Variables predictoras** (" + str(len(D.PREDICTORES)) + ")")
        st.write(
            "Categórica: `Sector` (One-Hot). Derivadas de la fecha: `Mes`, `Trimestre`. "
            "Numéricas: " + ", ".join(f"`{c}`" for c in D.NUMERICAS_ORIGINALES) + "."
        )
    with c2:
        st.markdown("**Variables excluidas**")
        st.write(
            "- `ID_Observacion`: identificador sin contenido predictivo.\n"
            "- `Prob_Tension_Liquidez`: probabilidad con la que se simuló el target (fuga de información).\n"
            "- `Brecha_Caja_90d_MXN`: resultado futuro, target del modelo de regresión (fuga de información).\n"
            "- `Fecha`: se usa para ordenar y para derivar Mes/Trimestre, no como columna cruda."
        )

    st.subheader("Vista previa de la base")
    st.dataframe(df.head(50), width="stretch", height=320)
    with st.expander("Estadísticos descriptivos"):
        st.dataframe(df[D.NUMERICAS_ORIGINALES].describe().T, width="stretch")

# ---------------------------------------------------------------------------
# 2. Evaluación
# ---------------------------------------------------------------------------
with tabs[1]:
    m_train = M.metricas(y_train, p_train, umbral)
    tabla = pd.DataFrame({"Entrenamiento": m_train, "Prueba": m_test})
    tabla["Diferencia"] = tabla["Entrenamiento"] - tabla["Prueba"]

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"Métricas con umbral {umbral:.2f}")
        st.dataframe(tabla.style.format("{:.3f}"), width="stretch")
    with c2:
        st.subheader("Diagnóstico")
        rf = pipe.named_steps["rf"]
        if params["bootstrap"] and hasattr(rf, "oob_score_"):
            st.metric("Accuracy out-of-bag (OOB)", formato_pct(rf.oob_score_),
                      help="Estimación interna con las observaciones que cada árbol no vio.")
        brecha = m_train["ROC-AUC"] - m_test["ROC-AUC"]
        if brecha > 0.10:
            st.warning(
                f"ROC-AUC de entrenamiento supera a la de prueba por {brecha:.2f}: posible sobreajuste. "
                "Pruebe limitar max_depth o subir min_samples_leaf."
            )
        else:
            st.success(f"Brecha ROC-AUC entrenamiento vs prueba: {brecha:.2f}")
        st.caption(
            "En Random Forest es normal que las métricas de entrenamiento sean cercanas a 100%; "
            "la referencia para decidir es el conjunto de prueba."
        )

    c1, c2 = st.columns(2)
    c1.plotly_chart(G.matriz_confusion(M.matriz_confusion(y_train, p_train, umbral), "Matriz de confusión — entrenamiento"),
                    width="stretch")
    c2.plotly_chart(G.matriz_confusion(M.matriz_confusion(y_test, p_test, umbral), "Matriz de confusión — prueba"),
                    width="stretch")

    c1, c2 = st.columns(2)
    c1.plotly_chart(G.curva_roc(y_train, p_train, y_test, p_test, m_train["ROC-AUC"], m_test["ROC-AUC"], umbral),
                    width="stretch")
    c2.plotly_chart(G.curva_pr(y_train, p_train, y_test, p_test, m_train["PR-AUC"], m_test["PR-AUC"], umbral),
                    width="stretch")

    c1, c2 = st.columns(2)
    c1.plotly_chart(G.metricas_vs_umbral(y_test, p_test, umbral), width="stretch")
    c2.plotly_chart(G.distribucion_probabilidades(p_test, y_test, umbral), width="stretch")

# ---------------------------------------------------------------------------
# 3. Importancia de variables
# ---------------------------------------------------------------------------
with tabs[2]:
    imp_imp = M.importancia_impureza(pipe)
    imp_perm = perm_cache(clave, pipe, X_test, y_test)
    top = st.slider("Variables a mostrar", 5, 30, 15, key="top_imp")
    c1, c2 = st.columns(2)
    c1.plotly_chart(G.barras_importancia(imp_imp, "Importancia por impureza (Gini/entropía)", top),
                    width="stretch")
    c2.plotly_chart(G.barras_importancia(imp_perm, "Importancia por permutación (caída de ROC-AUC en prueba)", top,
                                         con_error=True), width="stretch")
    st.caption(
        "La importancia por impureza se calcula en entrenamiento y tiende a favorecer variables continuas con "
        "muchos valores. La de permutación mide cuánto empeora el ROC-AUC de prueba al desordenar cada "
        "variable; es más confiable para interpretar."
    )
    with st.expander("Tablas de importancia"):
        c1, c2 = st.columns(2)
        c1.dataframe(imp_imp, hide_index=True, width="stretch")
        c2.dataframe(imp_perm, hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 4. SHAP
# ---------------------------------------------------------------------------
with tabs[3]:
    exp = shap_cache(clave, pipe, X_test)
    st.subheader("Efecto global de las variables (conjunto de prueba)")
    st.caption(
        "Cada punto es una observación. A la derecha del cero la variable aumenta la probabilidad de tensión; "
        "el color indica si el valor de la variable es alto (rojo) o bajo (azul)."
    )
    col_shap, _ = st.columns([3, 1])
    col_shap.pyplot(G.shap_resumen(exp), width="stretch")

    st.subheader("Explicación de una observación")
    opciones = list(range(len(X_test)))
    ids = df_test["ID_Observacion"].tolist() if "ID_Observacion" in df_test else opciones

    def etiqueta(i: int) -> str:
        return (f"ID {ids[i]} · {df_test[D.COL_FECHA].iloc[i].date()} · {df_test['Sector'].iloc[i]} · "
                f"real {int(y_test.iloc[i])} · prob {p_test[i]:.2f}")

    i = st.selectbox("Observación de prueba", opciones, format_func=etiqueta)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(G.gauge_probabilidad(float(p_test[i]), umbral), width="stretch")
        pred = int(p_test[i] >= umbral)
        st.write(f"Predicción: **{'Con tensión' if pred else 'Sin tensión'}** · Real: "
                 f"**{'Con tensión' if y_test.iloc[i] else 'Sin tensión'}**")
    with c2:
        st.pyplot(G.shap_cascada(exp[i]), width="stretch")

# ---------------------------------------------------------------------------
# 5. Predicción individual
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Evaluar una empresa")
    st.caption(
        "Valores iniciales = mediana del conjunto de entrenamiento. Los porcentajes se capturan como fracción "
        "(0.15 = 15%). CCC, capital de trabajo neto (NWC) y EBITDA se calculan automáticamente."
    )
    med = X_train[D.NUMERICAS_ORIGINALES].median()
    sectores = sorted(X_train["Sector"].astype(str).unique())

    with st.form("form_individual"):
        c1, c2, c3 = st.columns(3)
        sector = c1.selectbox("Sector", sectores)
        fecha = c2.date_input("Fecha de evaluación", value=date.today())
        estac = c3.selectbox("¿Negocio estacional?", [0, 1], index=int(med["Estacionalidad_bin"] >= 0.5),
                             format_func=lambda v: "Sí (1)" if v else "No (0)")

        valores = {}
        grupos = [
            ("Ventas y rentabilidad", ["Ventas_12M_MXN", "Crecimiento_Ventas_pct", "Margen_EBITDA_pct",
                                       "Volatilidad_Ventas_pct"]),
            ("Ciclo de conversión", ["DSO_dias", "DIO_dias", "DPO_dias"]),
            ("Balance de capital de trabajo", ["CxC_MXN", "Inventarios_MXN", "CxP_MXN"]),
            ("Liquidez y deuda", ["Caja_MXN", "Linea_Credito_Disponible_MXN", "Deuda_CP_MXN"]),
            ("Calidad de cartera e inventario", ["Concentracion_Top5_Clientes_pct", "Morosidad_CxC_pct",
                                                 "Inventario_Obsoleto_pct"]),
        ]
        for titulo, cols in grupos:
            st.markdown(f"**{titulo}**")
            cc = st.columns(len(cols))
            for c, var in zip(cc, cols):
                es_frac = var.endswith("_pct")
                valores[var] = c.number_input(
                    ETIQUETAS[var],
                    value=float(round(med[var], 4 if es_frac else 0)),
                    step=0.01 if es_frac else (1.0 if var.endswith("_dias") else 10000.0),
                    format="%.4f" if es_frac else "%.0f",
                )
        enviar = st.form_submit_button("Calcular probabilidad", type="primary")

    if enviar:
        fila = dict(valores)
        fila["Sector"] = sector
        fila["Estacionalidad_bin"] = estac
        fila["CCC_dias"] = fila["DSO_dias"] + fila["DIO_dias"] - fila["DPO_dias"]
        fila["NWC_MXN"] = fila["CxC_MXN"] + fila["Inventarios_MXN"] - fila["CxP_MXN"]
        fila["EBITDA_MXN"] = fila["Ventas_12M_MXN"] * fila["Margen_EBITDA_pct"]
        fila["Fecha"] = pd.Timestamp(fecha)
        X_nuevo, _ = D.preparar_para_prediccion(pd.DataFrame([fila]))
        prob = float(M.probabilidades(pipe, X_nuevo)[0])

        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(G.gauge_probabilidad(prob, umbral), width="stretch")
            if prob >= umbral:
                st.error(f"**Con tensión de liquidez** (probabilidad {prob:.1%} ≥ umbral {umbral:.2f})")
            else:
                st.success(f"**Sin tensión de liquidez** (probabilidad {prob:.1%} < umbral {umbral:.2f})")
            st.caption(f"Calculados: CCC {fila['CCC_dias']:.1f} días · NWC ${fila['NWC_MXN']:,.0f} · "
                       f"EBITDA ${fila['EBITDA_MXN']:,.0f}")
        with c2:
            exp_n = M.explicacion_shap(pipe, X_nuevo)
            st.pyplot(G.shap_cascada(exp_n[0]), width="stretch")

# ---------------------------------------------------------------------------
# 6. Predicción masiva
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Predicción para varias empresas")
    st.write(
        "Suba un Excel o CSV con las columnas predictoras y `Fecha` (o `Mes`). "
        "Si el archivo incluye `Tension_Liquidez_bin`, también se calculan métricas."
    )
    plantilla = df_test[[D.COL_FECHA] + D.CATEGORICAS + D.NUMERICAS_ORIGINALES].head(10)
    st.download_button("Descargar plantilla (10 filas de ejemplo)", a_excel(plantilla, "Datos_Modelo"),
                       "plantilla_prediccion.xlsx")

    lote = st.file_uploader("Archivo para predecir", type=["xlsx", "xls", "csv"], key="lote")
    if lote is not None:
        try:
            df_lote = D.leer_archivo(lote)
            X_lote, falt = D.preparar_para_prediccion(df_lote)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el archivo: {e}")
            X_lote, falt = None, ["(archivo ilegible)"]
        if falt:
            st.error("Faltan columnas: " + ", ".join(falt))
        else:
            nulos = X_lote.isna().any(axis=1)
            if nulos.any():
                st.warning(f"{int(nulos.sum())} fila(s) con valores vacíos o no numéricos se omitieron.")
            prob_lote = np.full(len(X_lote), np.nan)
            if (~nulos).any():
                prob_lote[~nulos.values] = M.probabilidades(pipe, X_lote[~nulos])
            resultado = df_lote.copy()
            resultado["Prob_Tension_Estimada"] = prob_lote
            resultado["Prediccion_Tension"] = np.where(np.isnan(prob_lote), np.nan, (prob_lote >= umbral).astype(float))
            resultado["Umbral_Usado"] = umbral

            validos = ~np.isnan(prob_lote)
            c1, c2, c3 = st.columns(3)
            c1.metric("Filas evaluadas", int(validos.sum()))
            c2.metric("Con tensión", int(np.nansum(resultado["Prediccion_Tension"])))
            c3.metric("% con tensión", formato_pct(np.nanmean(resultado["Prediccion_Tension"])))

            if D.TARGET in df_lote.columns and validos.any():
                y_lote = pd.to_numeric(df_lote[D.TARGET], errors="coerce")
                ok = validos & y_lote.notna().values
                if ok.sum() > 0 and y_lote[ok].nunique() > 1:
                    ml = M.metricas(y_lote[ok].astype(int), prob_lote[ok], umbral)
                    st.dataframe(pd.DataFrame([ml]).style.format("{:.3f}"), hide_index=True,
                                 width="stretch")

            st.dataframe(resultado, width="stretch", height=360)
            c1, c2 = st.columns(2)
            c1.download_button("Descargar resultados (Excel)", a_excel(resultado), "predicciones_tension.xlsx",
                               type="primary")
            c2.download_button("Descargar resultados (CSV)", resultado.to_csv(index=False).encode("utf-8-sig"),
                               "predicciones_tension.csv", "text/csv")

# ---------------------------------------------------------------------------
# 7. Exportar modelo
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Descargar el modelo entrenado")
    metadatos = {
        "umbral": umbral,
        "parametros": params,
        "predictores": D.PREDICTORES,
        "target": D.TARGET,
        "metricas_prueba": m_test,
        "particion": f"temporal {PCT_ENTRENAMIENTO:.0%}/{1 - PCT_ENTRENAMIENTO:.0%}",
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "version_sklearn": sklearn.__version__,
    }
    st.download_button(
        "Descargar modelo (.joblib)", M.serializar(pipe, metadatos), "rf_tension_liquidez.joblib",
        type="primary",
    )
    st.json({k: v for k, v in metadatos.items() if k != "predictores"}, expanded=False)
    st.markdown("**Cómo usarlo fuera de la app**")
    st.code(
        f"""import joblib, pandas as pd

paquete = joblib.load("rf_tension_liquidez.joblib")
pipe, umbral = paquete["pipeline"], paquete["umbral"]

df = pd.read_excel("nuevas_empresas.xlsx")
fecha = pd.to_datetime(df["Fecha"])
df["Mes"], df["Trimestre"] = fecha.dt.month, fecha.dt.quarter

prob = pipe.predict_proba(df[paquete["predictores"]])[:, 1]
df["Prediccion_Tension"] = (prob >= umbral).astype(int)
# Requiere scikit-learn {sklearn.__version__} para cargar el modelo sin advertencias.
""",
        language="python",
    )
