"""Gráficas del modelo (Plotly para interactivas, Matplotlib para SHAP)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve

# Paleta categórica validada (modo claro): slot 1 azul, slot 2 naranja
AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS = "#8a8984"
TEXTO_SEC = "#52514e"
ESCALA_AZUL = [[0.0, "#eef4fc"], [1.0, "#1b4f8f"]]

LAYOUT_BASE = dict(
    margin=dict(l=10, r=10, t=50, b=10),
    font=dict(size=13),
    hoverlabel=dict(font_size=12),
)


def matriz_confusion(cm: np.ndarray, titulo: str) -> go.Figure:
    etiquetas = ["Sin tensión (0)", "Con tensión (1)"]
    total = cm.sum()
    texto = [[f"{v}<br>({v / total:.0%})" if total else str(v) for v in fila] for fila in cm]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=etiquetas,
            y=etiquetas,
            colorscale=ESCALA_AZUL,
            showscale=False,
            text=texto,
            texttemplate="%{text}",
            textfont=dict(size=15),
            xgap=2,
            ygap=2,
            hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Casos: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Predicho",
        yaxis_title="Real",
        yaxis=dict(autorange="reversed"),
        height=360,
        **LAYOUT_BASE,
    )
    return fig


def curva_roc(y_train, p_train, y_test, p_test, auc_train, auc_test, umbral) -> go.Figure:
    fig = go.Figure()
    for y, p, nombre, color, auc in [
        (y_train, p_train, "Entrenamiento", AZUL, auc_train),
        (y_test, p_test, "Prueba", NARANJA, auc_test),
    ]:
        fpr, tpr, thr = roc_curve(y, p)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{nombre} (AUC {auc:.3f})",
            line=dict(color=color, width=2),
            customdata=np.clip(thr, 0, 1),
            hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<br>Umbral %{customdata:.2f}<extra>" + nombre + "</extra>",
        ))
    # Punto del umbral actual sobre la curva de prueba
    fpr, tpr, thr = roc_curve(y_test, p_test)
    i = int(np.argmin(np.abs(thr - umbral)))
    fig.add_trace(go.Scatter(
        x=[fpr[i]], y=[tpr[i]], mode="markers", name=f"Umbral {umbral:.2f} (prueba)",
        marker=dict(size=11, color=NARANJA, line=dict(color="white", width=2)),
        hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra>Umbral actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Azar",
        line=dict(color=GRIS, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.update_layout(
        title="Curva ROC",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Tasa de verdaderos positivos (recall)",
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        **LAYOUT_BASE,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def curva_pr(y_train, p_train, y_test, p_test, ap_train, ap_test, umbral) -> go.Figure:
    fig = go.Figure()
    for y, p, nombre, color, ap in [
        (y_train, p_train, "Entrenamiento", AZUL, ap_train),
        (y_test, p_test, "Prueba", NARANJA, ap_test),
    ]:
        prec, rec, thr = precision_recall_curve(y, p)
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=f"{nombre} (AP {ap:.3f})",
            line=dict(color=color, width=2),
            customdata=np.append(thr, np.nan),
            hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<br>Umbral %{customdata:.2f}<extra>" + nombre + "</extra>",
        ))
    prec, rec, thr = precision_recall_curve(y_test, p_test)
    i = int(np.argmin(np.abs(thr - umbral)))
    fig.add_trace(go.Scatter(
        x=[rec[i]], y=[prec[i]], mode="markers", name=f"Umbral {umbral:.2f} (prueba)",
        marker=dict(size=11, color=NARANJA, line=dict(color="white", width=2)),
        hovertemplate="Recall %{x:.2f}<br>Precision %{y:.2f}<extra>Umbral actual</extra>",
    ))
    base = float(np.mean(y_test))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[base, base], mode="lines", name=f"Proporción base ({base:.0%})",
        line=dict(color=GRIS, width=1, dash="dot"), hoverinfo="skip",
    ))
    fig.update_layout(
        title="Curva Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=420,
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        **LAYOUT_BASE,
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def metricas_vs_umbral(y_test, p_test, umbral) -> go.Figure:
    """Precision, recall y F1 de prueba en función del umbral."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    umbrales = np.round(np.arange(0.05, 0.96, 0.01), 2)
    filas = []
    for u in umbrales:
        pred = (p_test >= u).astype(int)
        filas.append((
            precision_score(y_test, pred, zero_division=0),
            recall_score(y_test, pred, zero_division=0),
            f1_score(y_test, pred, zero_division=0),
        ))
    arr = np.array(filas)
    fig = go.Figure()
    for j, (nombre, color, dash) in enumerate([
        ("Precision", AZUL, "solid"), ("Recall", NARANJA, "solid"), ("F1", "#1baf7a", "dash")
    ]):
        fig.add_trace(go.Scatter(
            x=umbrales, y=arr[:, j], mode="lines", name=nombre,
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="Umbral %{x:.2f}<br>" + nombre + " %{y:.3f}<extra></extra>",
        ))
    fig.add_vline(x=umbral, line=dict(color=GRIS, width=1, dash="dot"),
                  annotation_text=f"Umbral {umbral:.2f}", annotation_position="top")
    fig.update_layout(
        title="Métricas de prueba según el umbral",
        xaxis_title="Umbral de decisión", yaxis_title="Valor",
        hovermode="x unified", height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        **LAYOUT_BASE,
    )
    fig.update_yaxes(range=[0, 1.02])
    return fig


def barras_importancia(df: pd.DataFrame, titulo: str, top: int = 15, con_error: bool = False) -> go.Figure:
    d = df.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["Importancia"],
        y=d["Variable"],
        orientation="h",
        marker=dict(color=AZUL, cornerradius=4),
        error_x=dict(type="data", array=d["Desv_Est"], color=GRIS, thickness=1) if con_error and "Desv_Est" in d else None,
        hovertemplate="%{y}<br>Importancia %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=titulo, xaxis_title="Importancia", height=max(360, 28 * len(d) + 90),
        bargap=0.25, **LAYOUT_BASE,
    )
    return fig


def distribucion_probabilidades(p_test, y_test, umbral) -> go.Figure:
    fig = go.Figure()
    for clase, nombre, color in [(0, "Real: sin tensión", AZUL), (1, "Real: con tensión", NARANJA)]:
        fig.add_trace(go.Histogram(
            x=p_test[np.asarray(y_test) == clase], name=nombre, marker_color=color,
            opacity=0.75, xbins=dict(start=0, end=1, size=0.05),
            hovertemplate="Prob. %{x}<br>Casos %{y}<extra>" + nombre + "</extra>",
        ))
    fig.add_vline(x=umbral, line=dict(color=GRIS, width=1, dash="dot"),
                  annotation_text=f"Umbral {umbral:.2f}", annotation_position="top")
    fig.update_layout(
        barmode="overlay", title="Distribución de probabilidades (prueba)",
        xaxis_title="Probabilidad estimada de tensión", yaxis_title="Casos", height=380,
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0),
        **LAYOUT_BASE,
    )
    return fig


def gauge_probabilidad(prob: float, umbral: float) -> go.Figure:
    color = NARANJA if prob >= umbral else AZUL
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix=" %", valueformat=".1f"),
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=color),
            threshold=dict(line=dict(color=TEXTO_SEC, width=3), thickness=0.8, value=umbral * 100),
        ),
        title=dict(text="Probabilidad de tensión de liquidez"),
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def shap_resumen(exp, max_display: int = 15):
    import shap

    plt.close("all")
    shap.plots.beeswarm(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 0.42 * max_display + 1.5)
    fig.tight_layout()
    return fig


def shap_cascada(exp_fila, max_display: int = 12):
    import shap

    plt.close("all")
    shap.plots.waterfall(exp_fila, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 0.45 * max_display + 1.5)
    fig.tight_layout()
    return fig
