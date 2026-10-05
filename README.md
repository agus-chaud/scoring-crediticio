# Credit Risk Scoring

Proyecto de ciencia de datos para estimar el riesgo de crédito a partir de información histórica de préstamos.  
Implementa tres modelos complementarios: probabilidad de incumplimiento (PD), exposición al incumplimiento (EAD) y pérdida dado el incumplimiento (LGD).  
Los modelos se exponen con una API FastAPI y se consumen desde un dashboard Streamlit.  
El resultado es la pérdida esperada, `PD × EAD × LGD`: una métrica útil para análisis, no una decisión crediticia, porque no tenemos los datos de costo de los falsos positivos ni de beneficio de los verdaderos positivos.

Diccionario de campos: [`Diccionario.xlsx`](Diccionario.xlsx) · Decisiones técnicas: [`decisions.md`](decisions.md) · Diseño de transformaciones: [`01_Documentos/Diseño_Transformaciones.md`](01_Documentos/Diseño_Transformaciones.md)

## Dashboard en vivo

Ver el dashboard en vivo: [https://aa-scoring-crediticio.onrender.com](https://aa-scoring-crediticio.onrender.com).
Tenele paciencia que cargue :)

## Hallazgos clave

- El modelo de PD ordena el riesgo con **AUC 0,708 (Gini 0,42)** sobre 35.592 préstamos de validación externa que no se usaron para entrenar. El valor coincide con la validación cruzada (0,707): no hay sobreajuste.
- La probabilidad está **calibrada**: la PD media predicha es **19,8%** frente a una tasa real de **19,7%**, y en cada decil la diferencia es de **±3,0 p.p.** como máximo. Se puede usar tal cual en la pérdida esperada.
- Revisando el **20%** de clientes con mayor PD se encuentra el **39%** de los impagos.
- El sector de empleo es la única variable fuera del contrato original que mejora el modelo: **+0,0036 de AUC**. Los clientes sin título de empleo tienen **26,6%** de impago, frente a 14–25% en el resto.
- EAD y LGD apenas superan a predecir siempre la mediana: MAE **0,159 frente a 0,177** en EAD (−9,8%) y **0,087 frente a 0,088** en LGD (−0,6%). El modelo de LGD asigna casi la misma pérdida (0,88–0,92) a todos los préstamos.
- `rating` y `tipo_interes` tienen correlación **0,95**: aportan casi la misma información, y con el escalado anterior el coeficiente de `tipo_interes` salía con el signo cambiado.

## Problema

Evaluar el riesgo de una cartera de préstamos requiere combinar la probabilidad de incumplimiento con la exposición pendiente y la pérdida potencial. Hay tres cosas que pueden sesgar el resultado si no se controlan: préstamos todavía vigentes etiquetados como buenos pagadores, transformaciones ajustadas con datos de validación (fuga de información) y probabilidades descalibradas, que inflan o subestiman la pérdida esperada aunque el modelo ordene bien.

## Objetivo

Construir un flujo reproducible de scoring crediticio que reciba 15 variables crudas de un préstamo, calcule PD, EAD, LGD y pérdida esperada relativa, y exponga el resultado mediante API y dashboard.

## Enfoque técnico

1. **Importación** (`01_ImportacionDatos`): separación de `validacion.pkl` antes de cualquier análisis.
2. **Calidad** (`02_Calidad de Datos`): eliminación de ingresos > 400.000, `dti = 999` y préstamos sin desenlace; imputaciones y recortes a límites de negocio.
3. **EDA** (`03_EDA`): definición de `target_pd` y tasa de impago por variable.
4. **Transformación** (`04_Transformacion de datos`): targets de EAD y LGD, y un `ColumnTransformer` persistido en `05_modelos/preprocesador.joblib`.
5. **Modelización** (`05`–`07`): PD con regresión logística; EAD y LGD con modelos de regresión entrenados solo sobre defaults.
6. **Producción** (`07_despliegue`): `01_reentrenamiento.py` serializa los tres pipelines en `artefacto_pipeline.pkl` → API FastAPI (`POST /predict`) → dashboard Streamlit.

Modelos en producción (artefacto 1.3.0): regresión logística Ridge (`l1_ratio=0`, `C≈0,316`) para PD, la mejor configuración de la búsqueda del notebook 05; `HistGradientBoostingRegressor` para EAD y LGD. El preprocesador de `01_reentrenamiento.py` replica el del notebook 04.

## Decisiones técnicas relevantes

### Un único preprocesador scikit-learn ([DEC-006](decisions.md))

Los encoders se ajustaban uno por uno, no se guardaban y las categorías raras se reagrupaban con `replace` de pandas: validación y producción no podían repetir la preparación. Ahora un `ColumnTransformer` guardado con joblib agrupa las categorías con menos de 200 casos, codifica las desconocidas sin fallar y aplica Yeo-Johnson a las numéricas asimétricas (la asimetría de `ingresos` bajó de 2,19 a 0,13). `01_reentrenamiento.py` usa la misma definición, así que las métricas de los notebooks describen el artefacto desplegado. Se usa `drop='first'` en One Hot porque, con k columnas, una se deduce de las demás y la regresión logística no tolera esa redundancia.

### Preprocesador ajustado dentro de la validación cruzada ([DEC-008](decisions.md))

Si el escalado se ajusta con todas las filas antes de partir los datos, el test influye en la preparación y la métrica sale optimista. El notebook 05 clona el preprocesador sin ajustar y lo mete en el `Pipeline`, así cada fold lo ajusta solo con su parte de entrenamiento. La partición es estratificada y con `random_state=42`.

### Métricas de ordenamiento y de calibración ([DEC-008](decisions.md))

El AUC dice si el modelo ordena bien, no si la probabilidad es correcta, y la pérdida esperada multiplica la probabilidad. Por eso el modelo se evalúa con AUC/Gini/KS (ordenamiento) y con Brier y curva de calibración (exactitud de la probabilidad), sobre el test interno y sobre `validacion.pkl`. La validación del despliegue (`03_validacion_externa.py`) aplica los mismos filtros de filas que el entrenamiento ([DEC-009](decisions.md)).

### Sector de empleo en el contrato de la API ([DEC-010](decisions.md))

De las cuatro variables con señal que quedaban fuera, solo `sector_empleo` mejora el modelo de forma apreciable en validación cruzada (+0,0036 de AUC; las otras tres, entre +0,0001 y +0,0007). La API recibe el título de empleo crudo (`empleo`, obligatorio pero admite `null`) y el pipeline lo convierte en sector con `04_scripts/sector_empleo.py`, la misma regla de la fase de calidad. El código del módulo viaja embebido en el artefacto (`cloudpickle.register_pickle_by_value`), así que la API no depende de `04_scripts`. Un cliente que siga mandando los 14 campos viejos recibe un error 422 explícito en lugar de quedar clasificado en silencio como `desconocido`, el sector de mayor riesgo.

### EAD y LGD comparados con un baseline ([DEC-011](decisions.md))

Los notebooks 06 y 07 reentrenaban el modelo final con hiperparámetros escritos a mano distintos del ganador de la búsqueda, partían sin semilla y no tenían referencia. Ahora usan `best_estimator_`, `random_state=42`, el preprocesador clonado dentro del `Pipeline` (como PD) y comparan el MAE con `DummyRegressor(strategy='median')`. Esa comparación mostró que el modelo de LGD casi no aprende.

### Sin rebalanceo de clases ([DEC-007](decisions.md))

La tasa de impago es 19,9% (16.568 casos): no hay escasez de positivos. Remuestrear desplazaría la PD hacia arriba y obligaría a recalibrarla antes de calcular la pérdida esperada.

### Contrato crudo en la API ([DEC-001](decisions.md), [DEC-003](decisions.md))

El cliente Streamlit no carga artefactos de ML ni transforma categorías o escalas; la API concentra todo el scoring. Así hay una sola implementación de la preparación en producción.

### Umbral demostrativo ([DEC-004](decisions.md))

El umbral de pérdida esperada relativa `≤ 0.05` es una referencia visual no calibrada. No representa una aprobación, oferta, política de riesgo ni asesoramiento financiero.

## Resultados principales

| Área analizada | Hallazgo | Interpretación |
|---|---|---|
| Ordenamiento PD | AUC 0,708 · Gini 0,42 · KS 0,31 en validación externa | Separa buenos y malos pagadores claramente mejor que el azar; el resultado es estable entre CV, test y validación externa. |
| Calibración PD | Brier 0,144 frente a 0,158 sin modelo; ±3,0 p.p. por decil | La probabilidad se puede usar directamente en `PD × EAD × LGD`. |
| Captura de impagos | 20% de mayor PD → 39% de los impagos | Priorizar la revisión por PD casi duplica la eficiencia frente a revisar al azar. |
| EAD | MAE 0,159 frente a 0,177 de la mediana (validación interna) | Mejora un 9,8%; las predicciones van de 0,55 a 0,95 y sobrestiman los EAD bajos. |
| LGD | MAE 0,087 frente a 0,088 de la mediana (validación interna) | Mejora un 0,6%: con las variables actuales el modelo no distingue préstamos. |
| Variables descartadas | `num_meses_desde_ult_retraso` y `num_cancelaciones_12meses`: AUC ≈ 0,50 | No aportan información y se excluyeron. |
| Sector de empleo | +0,0036 de AUC en CV; AUC externo 0,705 → 0,708 | Se incorporó al contrato de la API como título de empleo crudo. |
| Variables descartadas con señal | `num_hipotecas`, `porc_tarjetas_75p`, `tiene_descripcion`: AUC ≈ 0,55 solas, +0,0001 a +0,0007 dentro del modelo | Su información ya está en las otras variables; no justifican ampliar el contrato. |

## Funcionalidades entregadas

### Dashboard y visualizaciones

- Formulario con ocho variables editables: monto, cuotas, tasa, cuota, ingresos, DTI, uso revolving y rating.
- KPI de resultado con pérdida esperada relativa, banda de riesgo y métricas secundarias de PD, EAD y LGD.
- Gráfico de barra horizontal sobre una escala demostrativa de 0–25%, con referencias de 5% y 10%.
- Escenarios de sensibilidad *ceteris paribus*: exploran hasta seis alternativas que no incrementan el monto principal.
- Recuperación ante *cold starts* de la API mediante un calentamiento acotado de `/health` y un único reintento.

## Estructura del proyecto

```text
AA_scoring-crediticio/
├── 01_Documentos/
│   └── Diseño_Transformaciones.md        # Matriz de transformaciones, variable por variable
├── 02_datos/
│   ├── 01_Originales/prestamos.csv       # Datos fuente de préstamos
│   ├── 02_Validacion/validacion.pkl      # Validación externa, separada antes de entrenar
│   └── 03_Entrenamiento/                 # train.pkl y tablones derivados (calidad, EDA, PD, EAD, LGD)
├── 04_scripts/
│   └── sector_empleo.py                  # Título de empleo → sector (misma regla que Calidad)
├── 05_modelos/
│   └── preprocesador.joblib              # ColumnTransformer generado por el notebook 04
├── Notebooks/
│   ├── 01_ImportacionDatos.ipynb         # Carga y separación de validación
│   ├── 02_Calidad de Datos.ipynb         # Limpieza, imputación y filtros
│   ├── 03_EDA.ipynb                      # Target PD y análisis exploratorio
│   ├── 04_Transformacion de datos.ipynb  # Targets EAD/LGD y preprocesador
│   ├── 05_Modelizacion Clasificacion PD.ipynb  # Modelo PD, métricas de crédito y validación externa
│   ├── 06_Modelizacion Regresion EAD.ipynb     # Modelo EAD
│   ├── 07_Modelizacion Regresion LGD.ipynb     # Modelo LGD
│   └── 08_Preproduccion.ipynb            # Flujo consolidado
├── 06_resultados/Validacion/
│   ├── informe_validacion_modelos.md     # Informe de validación del artefacto
│   └── metricas_validacion_externa.json  # Métricas externas de PD, EAD y LGD
├── 07_despliegue/
│   ├── 01_reentrenamiento.py             # Entrena y genera el artefacto reproducible
│   ├── 03_validacion_externa.py          # Evalúa el artefacto sobre validacion.pkl
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
├── odd/tasks/                            # Seguimiento de tareas de cada mejora
├── decisions.md                          # Decisiones técnicas del proyecto
└── README.md                             # Este documento
```

## Tecnologías y dependencias

- **Python:** 3.13.7 para los despliegues configurados en Render.
- **Modelado y API:** FastAPI, Uvicorn, pandas, NumPy, scikit-learn, SciPy, joblib y cloudpickle.
- **Dashboard:** Streamlit, requests y Plotly.
- **Notebooks:** Jupyter, matplotlib, seaborn y python-dotenv.
- **Modelos:** `LogisticRegression`, `HistGradientBoostingRegressor`, `Pipeline` y `ColumnTransformer` de scikit-learn.

## Datos y artefactos

- **Origen:** `02_datos/01_Originales/prestamos.csv` y tablas derivadas en `02_datos/03_Entrenamiento/`.
- **Variables de entrada:** `ingresos_verificados`, `vivienda`, `finalidad`, `num_cuotas`, `antigüedad_empleo`, `rating`, `ingresos`, `dti`, `num_lineas_credito`, `porc_uso_revolving`, `principal`, `tipo_interes`, `imp_cuota`, `num_derogatorios` y `empleo` (título de empleo crudo; `null` si no se conoce).
- **Preprocesador de los notebooks:** `05_modelos/preprocesador.joblib` (15 variables → 40 columnas; en los notebooks entra `sector_empleo` ya calculado).
- **Clasificación de empleo:** `04_scripts/sector_empleo.py`, compartido por los notebooks y el reentrenamiento.
- **Artefacto de producción:** `07_despliegue/artefacto_pipeline.pkl`, con los tres pipelines y sus métricas de evaluación.
- **Salidas:** `score_pd`, `score_ead`, `score_lgd` y `perdida_esperada_relativa`.

## Instalación y ejecución local

> Requiere Python 3.13 o una versión compatible con las dependencias fijadas. Los datos (`02_datos/`) y los artefactos `.pkl`/`.joblib` no se versionan, salvo el artefacto de la API.

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

### Notebooks de transformación y modelización

No hay un archivo de dependencias para los notebooks: se instalan aparte. El notebook 04 genera `preprocesador.joblib` y los tablones; el 05, 06 y 07 los necesitan, así que el orden importa. El 06 y el 07 tardan unos 10 minutos cada uno. Las rutas son relativas a `Notebooks/` y los nombres de archivo tienen espacios.

```powershell
python -m pip install jupyter matplotlib seaborn python-dotenv

cd Notebooks
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace "04_Transformacion de datos.ipynb"
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace "05_Modelizacion Clasificacion PD.ipynb"
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace "06_Modelizacion Regresion EAD.ipynb"
..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace "07_Modelizacion Regresion LGD.ipynb"
```

### API y dashboard

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

## Limitaciones y próximos pasos

- **Clasificación de empleo por palabras clave.** El 18,5% de los clientes de entrenamiento queda en `otros` porque su título no coincide con ningún sector. La app de demostración envía siempre "Office Manager" (`administrativo`, 19,6% de impago, la tasa más cercana al promedio) como campo oculto.
- **Colinealidad sin resolver.** `rating`/`tipo_interes` (0,95) y `principal`/`imp_cuota` (0,97) siguen juntas en el modelo; falta la fase de selección de variables.
- **Clientes repetidos entre entrenamiento y validación.** Ningún préstamo se repite, pero 41.895 `id_cliente` de validación aparecen también en entrenamiento. Si un mismo cliente tiene varios préstamos, la validación puede ser algo optimista.
- **LGD sin poder predictivo.** El modelo mejora un 0,6% sobre la mediana. Falta probar otras variables o un enfoque en dos etapas (clasificar si la LGD es 1 y luego hacer una regresión para el resto).
- **Producción desalineada en EAD y LGD.** `01_reentrenamiento.py` y `08_Preproduccion.ipynb` no incorporan las correcciones de [DEC-011](decisions.md).
- **Sin umbral de decisión.** Falta cuantificar el costo de los falsos positivos y el beneficio de los verdaderos positivos para elegir un umbral que maximice el valor esperado.

## Notas de seguridad y uso responsable

- No subas credenciales, variables de entorno ni datos sensibles al repositorio.
- El dashboard es demostrativo: sus resultados no deben utilizarse como aprobación, denegación, oferta, *pricing* ni recomendación crediticia.
- PD, EAD y LGD son estimaciones de modelo; EAD y LGD representan proporciones condicionadas al incumplimiento, no montos monetarios absolutos.
