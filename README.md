# Random Forest · Predicción de tensión de liquidez

Aplicación **Streamlit** que entrena y evalúa un modelo **Random Forest de clasificación binaria** para predecir
`Tension_Liquidez_bin` a partir de indicadores de capital de trabajo (DSO, DIO, DPO, CCC, caja, deuda de corto
plazo, morosidad, etc.).

> Base didáctica con datos 100% simulados (600 observaciones, 2021-2026). No representa una empresa real.

## Configuración del modelo

| Parámetro | Valor |
|---|---|
| Objetivo | `Tension_Liquidez_bin` (1 = hay tensión de liquidez) |
| Algoritmo | `RandomForestClassifier` en un `Pipeline` de scikit-learn |
| Predictores | 24: `Sector` (One-Hot) + 20 numéricas + `Mes` y `Trimestre` derivados de `Fecha` |
| Excluidas | `ID_Observacion`, `Prob_Tension_Liquidez` y `Brecha_Caja_90d_MXN` (fuga de información), `Fecha` como columna cruda |
| Partición | **Temporal 80/20**: el 80% más antiguo entrena y el 20% más reciente se usa como prueba |
| Hiperparámetros | Ajustables con sliders: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `criterion`, `bootstrap` |
| Balance de clases | `class_weight='balanced'` |
| Umbral | Ajustable de 0.05 a 0.95 (default 0.50) |
| Semilla | `random_state=42` |

## Funciones de la app

1. **Datos y partición**: periodos de entrenamiento y prueba, proporción de clases y vista previa.
2. **Evaluación**: accuracy, precision, recall, F1, ROC-AUC y PR-AUC (entrenamiento contra prueba), OOB, matrices de
   confusión, curvas ROC y Precision-Recall, métricas según el umbral y distribución de probabilidades.
3. **Importancia de variables**: por impureza y por permutación (caída de ROC-AUC en prueba).
4. **SHAP**: gráfico beeswarm global y cascada (waterfall) para una observación.
5. **Predicción individual**: formulario por empresa. CCC, NWC y EBITDA se calculan automáticamente.
6. **Predicción masiva**: se sube un Excel/CSV y se descargan los resultados en Excel o CSV (incluye plantilla).
7. **Exportar modelo**: descarga del pipeline entrenado en `.joblib`, con umbral y metadatos.

## Estructura

```
.
├── app.py                  # Interfaz Streamlit
├── src/
│   ├── datos.py            # Carga, validación, variables de fecha y partición temporal
│   ├── modelo.py           # Pipeline, métricas, importancias, SHAP, exportación
│   └── graficas.py         # Gráficas Plotly y SHAP
├── data/
│   └── Base_Didactica_Random_Forest_Capital_Trabajo.xlsx
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

## Ejecutar en local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    ·    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Requiere Python 3.11 o superior.

## Subir a GitHub

```bash
git init
git add .
git commit -m "Random Forest tensión de liquidez - app Streamlit"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main
```

También puede crear el repositorio en github.com y arrastrar los archivos con **Add file → Upload files**
(incluya la carpeta `.streamlit`, que algunos exploradores ocultan).

## Publicar en Streamlit Community Cloud

1. Entre a <https://share.streamlit.io> con su cuenta de GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Seleccione el repositorio, la rama `main` y el archivo principal `app.py`.
4. En **Advanced settings** elija Python 3.12.
5. Presione **Deploy**. La primera instalación tarda unos minutos.

## Usar otra base

En la barra lateral, **Subir archivo (Excel/CSV)**. El archivo debe contener `Fecha`, `Sector`,
`Tension_Liquidez_bin` y las 20 variables numéricas de la base prototipo, con los mismos nombres. Si es Excel,
se lee la hoja `Datos_Modelo` (o la primera hoja si no existe). Los porcentajes van como fracción (0.15 = 15%).

## Usar el modelo exportado

```python
import joblib, pandas as pd

paquete = joblib.load("rf_tension_liquidez.joblib")
pipe, umbral = paquete["pipeline"], paquete["umbral"]

df = pd.read_excel("nuevas_empresas.xlsx")
fecha = pd.to_datetime(df["Fecha"])
df["Mes"], df["Trimestre"] = fecha.dt.month, fecha.dt.quarter

prob = pipe.predict_proba(df[paquete["predictores"]])[:, 1]
df["Prediccion_Tension"] = (prob >= umbral).astype(int)
```

Para cargar el `.joblib` use la misma versión de scikit-learn con la que se entrenó (se guarda en
`paquete["version_sklearn"]`).

## Notas metodológicas

- **Partición temporal.** Evita que el modelo aprenda con información posterior al periodo que se evalúa. Con
  120 observaciones de prueba, cambiar unos cuantos casos mueve las métricas varios puntos.
- **Métricas de entrenamiento.** En Random Forest suelen acercarse al 100%. Para decidir, use las de prueba y
  el OOB.
- **Umbral.** Bajarlo aumenta el recall (se detectan más empresas en tensión) a costa de más falsas alarmas. En
  este problema, un caso de tensión no detectado suele costar más que una falsa alarma.
- **Importancias.** La importancia por impureza favorece variables continuas; la de permutación y SHAP son más
  confiables para interpretar.
