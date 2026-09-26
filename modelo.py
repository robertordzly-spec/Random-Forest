"""Construcción, entrenamiento, evaluación e interpretación del Random Forest."""

from __future__ import annotations

import io

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from datos import CATEGORICAS, NUMERICAS

RANDOM_STATE = 42

PARAMETROS_DEFAULT = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "criterion": "gini",
    "bootstrap": True,
}


def construir_pipeline(params: dict) -> Pipeline:
    """Pipeline: One-Hot para Sector + Random Forest con class_weight='balanced'."""
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
            ("num", "passthrough", NUMERICAS),
        ],
        verbose_feature_names_out=False,
    )
    bootstrap = bool(params.get("bootstrap", True))
    modelo = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        criterion=params["criterion"],
        bootstrap=bootstrap,
        oob_score=bootstrap,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocesador), ("rf", modelo)])


def entrenar(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> Pipeline:
    pipe = construir_pipeline(params)
    pipe.fit(X_train, y_train)
    return pipe


def probabilidades(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilidad de la clase 1 (tensión de liquidez)."""
    return pipe.predict_proba(X)[:, 1]


def clasificar(prob: np.ndarray, umbral: float) -> np.ndarray:
    return (prob >= umbral).astype(int)


def metricas(y_true, prob, umbral: float) -> dict:
    """Métricas de clasificación para un umbral dado."""
    y_pred = clasificar(prob, umbral)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": auc,
        "PR-AUC": average_precision_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
    }


def matriz_confusion(y_true, prob, umbral: float) -> np.ndarray:
    return confusion_matrix(y_true, clasificar(prob, umbral), labels=[0, 1])


def nombres_transformados(pipe: Pipeline) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def importancia_impureza(pipe: Pipeline) -> pd.DataFrame:
    """Importancia por reducción de impureza (feature_importances_)."""
    imp = pipe.named_steps["rf"].feature_importances_
    return (
        pd.DataFrame({"Variable": nombres_transformados(pipe), "Importancia": imp})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def importancia_permutacion(pipe: Pipeline, X_test, y_test, n_repeats: int = 10) -> pd.DataFrame:
    """Importancia por permutación sobre el conjunto de prueba (caída en ROC-AUC)."""
    r = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.DataFrame({
            "Variable": list(X_test.columns),
            "Importancia": r.importances_mean,
            "Desv_Est": r.importances_std,
        })
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def explicacion_shap(pipe: Pipeline, X: pd.DataFrame):
    """Valores SHAP (clase 1) con TreeExplainer. Devuelve un shap.Explanation."""
    import shap

    X_t = pd.DataFrame(pipe.named_steps["prep"].transform(X), columns=nombres_transformados(pipe), index=X.index)
    explainer = shap.TreeExplainer(pipe.named_steps["rf"])
    exp = explainer(X_t, check_additivity=False)
    # En clasificación binaria SHAP devuelve (n, variables, clases): tomamos la clase 1
    if exp.values.ndim == 3:
        base = exp.base_values[:, 1] if np.ndim(exp.base_values) == 2 else exp.base_values
        exp = shap.Explanation(
            values=exp.values[:, :, 1],
            base_values=base,
            data=exp.data,
            feature_names=exp.feature_names,
        )
    return exp


def serializar(pipe: Pipeline, metadatos: dict) -> bytes:
    """Serializa el pipeline + metadatos (umbral, parámetros, predictores) a bytes .joblib."""
    buffer = io.BytesIO()
    joblib.dump({"pipeline": pipe, **metadatos}, buffer)
    return buffer.getvalue()
