"""Carga, validación y preparación de datos para el modelo de tensión de liquidez."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUTA_BASE = Path(__file__).resolve().parent / "data" / "Base_Didactica_Random_Forest_Capital_Trabajo.xlsx"
HOJA_DATOS = "Datos_Modelo"

TARGET = "Tension_Liquidez_bin"
COL_FECHA = "Fecha"

# Variables excluidas del entrenamiento (identificador y variables con fuga de información)
EXCLUIDAS = ["ID_Observacion", "Prob_Tension_Liquidez", "Brecha_Caja_90d_MXN"]

CATEGORICAS = ["Sector"]

NUMERICAS_ORIGINALES = [
    "Ventas_12M_MXN",
    "Crecimiento_Ventas_pct",
    "Margen_EBITDA_pct",
    "DSO_dias",
    "DIO_dias",
    "DPO_dias",
    "CCC_dias",
    "CxC_MXN",
    "Inventarios_MXN",
    "CxP_MXN",
    "NWC_MXN",
    "Deuda_CP_MXN",
    "Caja_MXN",
    "Linea_Credito_Disponible_MXN",
    "Concentracion_Top5_Clientes_pct",
    "Morosidad_CxC_pct",
    "Inventario_Obsoleto_pct",
    "Volatilidad_Ventas_pct",
    "Estacionalidad_bin",
    "EBITDA_MXN",
]

# Variables derivadas de la fecha
DERIVADAS_FECHA = ["Mes", "Trimestre"]

NUMERICAS = NUMERICAS_ORIGINALES + DERIVADAS_FECHA
PREDICTORES = CATEGORICAS + NUMERICAS

# Columnas mínimas que debe traer un archivo para entrenar
REQUERIDAS_ENTRENAMIENTO = [COL_FECHA, TARGET] + CATEGORICAS + NUMERICAS_ORIGINALES


def leer_archivo(origen) -> pd.DataFrame:
    """Lee un Excel (hoja Datos_Modelo si existe, si no la primera) o un CSV.

    `origen` puede ser una ruta o un archivo subido en Streamlit.
    """
    nombre = str(getattr(origen, "name", origen)).lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(origen)
    hojas = pd.ExcelFile(origen).sheet_names
    hoja = HOJA_DATOS if HOJA_DATOS in hojas else hojas[0]
    if hasattr(origen, "seek"):
        origen.seek(0)
    return pd.read_excel(origen, sheet_name=hoja)


def cargar_base(origen=None) -> pd.DataFrame:
    """Carga la base de entrenamiento: la del repositorio o la que suba el usuario."""
    return leer_archivo(origen if origen is not None else RUTA_BASE)


def validar_columnas(df: pd.DataFrame, requeridas: list[str]) -> list[str]:
    """Devuelve la lista de columnas faltantes (vacía si está completo)."""
    return [c for c in requeridas if c not in df.columns]


def agregar_variables_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Crea Mes y Trimestre a partir de Fecha (o Trimestre a partir de Mes)."""
    df = df.copy()
    if COL_FECHA in df.columns:
        fecha = pd.to_datetime(df[COL_FECHA], errors="coerce")
        df["Mes"] = fecha.dt.month
        df["Trimestre"] = fecha.dt.quarter
    elif "Mes" in df.columns:
        df["Mes"] = pd.to_numeric(df["Mes"], errors="coerce")
        df["Trimestre"] = ((df["Mes"] - 1) // 3 + 1)
    return df


def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia la base de entrenamiento: fechas, variables derivadas, orden cronológico."""
    df = df.copy()
    df[COL_FECHA] = pd.to_datetime(df[COL_FECHA], errors="coerce")
    df = df.dropna(subset=[COL_FECHA, TARGET])
    df[TARGET] = df[TARGET].astype(int)
    df = agregar_variables_fecha(df)
    df = df.sort_values(COL_FECHA, kind="stable").reset_index(drop=True)
    return df


def particion_temporal(df: pd.DataFrame, pct_entrenamiento: float = 0.80):
    """Partición temporal: el % más antiguo entrena, el más reciente se usa como prueba.

    Supone que `df` ya viene ordenado por fecha (ver `preparar_base`).
    """
    corte = int(round(len(df) * pct_entrenamiento))
    entrenamiento = df.iloc[:corte]
    prueba = df.iloc[corte:]
    X_train, y_train = entrenamiento[PREDICTORES], entrenamiento[TARGET]
    X_test, y_test = prueba[PREDICTORES], prueba[TARGET]
    return X_train, X_test, y_train, y_test, entrenamiento, prueba


def preparar_para_prediccion(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Prepara un archivo nuevo para predecir. Devuelve (X, columnas_faltantes)."""
    df = agregar_variables_fecha(df)
    faltantes = validar_columnas(df, PREDICTORES)
    if faltantes:
        return None, faltantes
    X = df[PREDICTORES].copy()
    for c in NUMERICAS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X["Sector"] = X["Sector"].astype(str)
    return X, []
