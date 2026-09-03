# Credit Risk Scoring

Proyecto de ciencia de datos para estimar el riesgo de crédito a partir de información histórica de préstamos.  
Implementa tres modelos complementarios: probabilidad de incumplimiento (PD), exposición al incumplimiento (EAD) y pérdida dado el incumplimiento (LGD).  
Los modelos se exponen con una API FastAPI y se consumen desde un dashboard Streamlit.  
El resultado es la pérdida esperada: `PD × EAD × LGD`, una métrica util para análisis y no una decisión crediticia, ya que no tenemos los datos necesarios de costo de falsos positivos y el beneficio de verdaderos positivos.

## Dashboard en vivo

Ver el dashboard en vivo: [https://aa-scoring-crediticio.onrender.com](https://aa-scoring-crediticio.onrender.com).

## Problema

Evaluar el riesgo de una cartera de préstamos necesita combinar la probabilidad de incumplimiento con la exposición pendiente y la pérdida potencial. Por eso se transforman registros crudos de préstamos en estimaciones consistentes de esas tres dimensiones.

## Objetivo

Construir un flujo reproducible de scoring crediticio que reciba 14 variables crudas de un préstamo, calcule PD, EAD, LGD y pérdida esperada relativa, y exponga el resultado mediante API y dashboard.

## Enfoque técnico

- **Metodología:** preparación determinística de variables, separación entrenamiento/validación con `random_state=42`, y preprocesamiento encapsulado dentro de cada `Pipeline` para evitar leakage. .
- **Modelos usados:** regresión logística con regularización L1 para PD; `HistGradientBoostingRegressor` para EAD y LGD.
- **Pipeline:** datos de entrenamiento → limpieza y transformación → entrenamiento/evaluación → serialización de `artefacto_pipeline.pkl` → API FastAPI (`POST /predict`) → dashboard Streamlit.

## Funcionalidades entregadas

### Dashboard y visualizaciones

- Formulario con ocho variables editables: monto, cuotas, tasa, cuota, ingresos, DTI, uso revolving y rating.
- Kpi de resultado con pérdida esperada relativa, banda de riesgo y métricas secundarias de PD, EAD y LGD.
- Gráfico de barra horizontal  sobre una escala demostrativa de 0–25%, con referencias de 5% y 10%.
- Escenarios de sensibilidad *ceteris paribus*: exploran hasta seis alternativas que no incrementan el monto principal.
- Recuperación ante *cold starts* de la API mediante un calentamiento acotado de `/health` y un único reintento.

### Decisiones técnicas relevantes

- **Reproducibilidad y leakage:** el artefacto productivo fija `random_state=42` y ajusta transformadores dentro del pipeline, a diferencia del flujo histórico de notebooks.
- **Contrato crudo:** el cliente Streamlit no carga artefactos de ML ni transforma categorías o escalas; la API  concentra todo el scoring.
- **Límite demostrativo:** el umbral de pérdida esperada relativa `≤ 0.05` es una referencia visual no calibrada. No representa una aprobación, oferta, política de riesgo ni asesoramiento financiero.
- **Mejora pendiente:** cuantificar el coste de falsos positivos y el beneficio de verdaderos positivos para seleccionar un umbral que maximice el valor esperado.

## Estructura del proyecto

```text
AA_scoring-crediticio/
├── 02_datos/
│   ├── 01_Originales/prestamos.csv       # Datos fuente de préstamos
│   └── 03_Entrenamiento/train.pkl        # Datos preparados para entrenamiento
├── Notebooks/                            # Flujo histórico: importación, calidad, EDA, transformación y modelos
├── 06_resultados/Validacion/
│   ├── informe_validacion_modelos.md     # Informe de validación
│   └── metricas_validacion_externa.json  # Métricas externas de PD, EAD y LGD
├── 07_despliegue/
│   ├── 01_reentrenamiento.py             # Entrena y genera el artefacto reproducible
│   ├── artefacto_pipeline.pkl            # Pipelines PD, EAD y LGD serializados
│   ├── api/
│   │   ├── main.py                       # API FastAPI: /health y /predict
│   │   ├── scoring.py                    # Motor de scoring
│   │   ├── requirements.txt              # Dependencias de la API
│   │   └── RENDER_DEPLOY.md              # Guía de despliegue de la API
│   └── app/
│       ├── app.py                        # Dashboard Streamlit
│       ├── app_core.py                   # Lógica del cliente y presentación
│       ├── requirements.txt              # Dependencias del dashboard
│       └── README.md                     # Instrucciones específicas de la app
├── decisions.md                          # Decisiones técnicas del proyecto
└── README.md                             # Este documento
```

## Tecnologías y dependencias

- **Python:** 3.13.7 para los despliegues configurados en Render.
- **Modelado y API:** FastAPI, Uvicorn, pandas, NumPy, scikit-learn, SciPy, joblib y cloudpickle.
- **Dashboard:** Streamlit, requests y Plotly.
- **Modelos:** `LogisticRegression`, `HistGradientBoostingRegressor`, `Pipeline` y `ColumnTransformer` de scikit-learn.

## Datos y artefactos

- **Origen:** `02_datos/01_Originales/prestamos.csv` y tablas derivadas almacenadas en `02_datos/03_Entrenamiento/`.
- **Variables de entrada:** `ingresos_verificados`, `vivienda`, `finalidad`, `num_cuotas`, `antigüedad_empleo`, `rating`, `ingresos`, `dti`, `num_lineas_credito`, `porc_uso_revolving`, `principal`, `tipo_interes`, `imp_cuota` y `num_derogatorios`.
- **Artefacto de producción:** `07_despliegue/artefacto_pipeline.pkl`, que contiene los tres pipelines y sus métricas de evaluación.
- **Salidas:** `score_pd`, `score_ead`, `score_lgd` y `perdida_esperada_relativa`.

## Instalación y ejecución local

> Requiere Python 3.13 o una versión compatible con las dependencias fijadas.

```powershell
# Crear y activar un entorno virtual (si aún no existe)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar dependencias de API y dashboard
python -m pip install -r 07_despliegue\api\requirements.txt
python -m pip install -r 07_despliegue\app\requirements.txt

# Opcional: regenerar el artefacto de modelado
python 07_despliegue\01_reentrenamiento.py
```

Iniciá la API en una terminal:

```powershell
cd 07_despliegue\api
..\..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --app-dir ..
```

En otra terminal, iniciá el dashboard:

```powershell
.venv\Scripts\python.exe -m streamlit run 07_despliegue\app\app.py
```

El dashboard usa por defecto `http://127.0.0.1:8000`. Para conectar una API desplegada, definí `API_BASE_URL` con su URL pública.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Comprueba el estado del servicio. |
| `POST` | `/predict` | Recibe una lista de registros crudos y devuelve las cuatro métricas de scoring. |
| `GET` | `/docs` | Documentación interactiva generada por FastAPI. |

## Notas de seguridad y uso responsable

- No subas credenciales, variables de entorno ni datos sensibles al repositorio.
- El dashboard es demostrativo: sus resultados no deben utilizarse como aprobación, denegación, oferta, *pricing* ni recomendación crediticia.
- PD, EAD y LGD son estimaciones de modelo; EAD y LGD representan proporciones condicionadas al incumplimiento, no montos monetarios absolutos.
