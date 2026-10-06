# Credit Risk Scoring: pérdida esperada (PD × EAD × LGD) sobre 200.000 préstamos

Modelos de probabilidad de impago (PD), exposición (EAD) y pérdida dado el impago (LGD) entrenados con `prestamos.csv` (200.000 préstamos históricos; el dataset no trae fecha así que no hay ventana temporal), servidos con FastAPI y consumidos desde un dashboard Streamlit: [dashboard en vivo](https://aa-scoring-crediticio.onrender.com) (el primer acceso puede tardar por el arranque en frío).

Diccionario de campos: [`Diccionario.xlsx`](Diccionario.xlsx) · Decisiones técnicas: [`decisions.md`](decisions.md) · Matriz de transformaciones: [`01_Documentos/Diseño_Transformaciones.md`](01_Documentos/Diseño_Transformaciones.md)

## Hallazgos clave

- **PD ordena bien y está calibrada.** AUC **0,708** (Gini 0,42) sobre **35.592** préstamos de validación externa no usados para entrenar; la validación cruzada da 0,707, sin sobreajuste. La PD media predicha es **19,8%** frente a una tasa real de **19,7%**.
- **Revisar el 20% de mayor PD captura el 39% de los impagos**, casi el doble que revisar al azar.
- **EAD y LGD apenas superan a predecir la mediana.** MAE **0,159 vs 0,177** en EAD (−9,8%) y **0,087 vs 0,088** en LGD (−0,6%). El modelo de LGD asigna casi la misma pérdida (0,88–0,92) a todos los préstamos.
- **El sector de empleo es la única variable fuera del contrato original que mejora el modelo:** +**0,0036** de AUC en CV. Los clientes sin título de empleo tienen **26,6%** de impago, frente a 14–25% en el resto.
- **`rating` y `tipo_interes` tienen correlación 0,95** (y `principal`/`imp_cuota`, 0,97): con el escalado anterior el coeficiente de `tipo_interes` salía con el signo invertido.

## Resumen ejecutivo

Proyecto para estimar el riesgo de crédito de una cartera de préstamos. Se construyó el flujo completo: limpieza, EDA, preprocesador reutilizable, tres modelos, artefacto serializado, API y dashboard. El resultado es la pérdida esperada `PD × EAD × LGD`, una métrica de análisis y no una decisión crediticia: faltan los datos de costo de los falsos positivos y de beneficio de los verdaderos positivos.

## Problema

Tres cosas pueden sesgar la pérdida esperada si no se controlan:

- **Préstamos vigentes etiquetados como buenos pagadores.** Un préstamo `Current` todavía no tiene desenlace; tratarlo como pagado subestima la PD.
- **Fuga de información.** Transformaciones ajustadas con datos de validación inflan la métrica.
- **Probabilidades descalibradas.** El modelo puede ordenar bien (AUC alto) y aun así inflar o subestimar la pérdida, porque ésta multiplica la probabilidad.

## Objetivo

Construir un flujo reproducible que reciba 15 variables crudas de un préstamo, calcule PD, EAD, LGD y pérdida esperada relativa, y los exponga por API y dashboard.

## Enfoque técnico

1. **Importación** (`01_ImportacionDatos`): se separa `validacion.pkl` antes de cualquier análisis.
2. **Calidad** (`02_Calidad de Datos`): se eliminan ingresos > 400.000, `dti = 999` y préstamos sin desenlace; imputaciones y recortes a límites de negocio.
3. **EDA** (`03_EDA`): definición de `target_pd` y tasa de impago por variable.
4. **Transformación** (`04_Transformacion de datos`): targets de EAD y LGD y un `ColumnTransformer` persistido en `05_modelos/preprocesador.joblib`.
5. **Modelización** (`05`–`07`): PD con regresión logística; EAD y LGD con `HistGradientBoostingRegressor` entrenado solo sobre impagos.
6. **Producción** (`07_despliegue`): `01_reentrenamiento.py` serializa los tres pipelines en `artefacto_pipeline.pkl` → API FastAPI (`POST /predict`) → dashboard Streamlit.

Modelos en producción (artefacto 1.3.0): regresión logística Ridge (`l1_ratio=0`, `C≈0,316`) para PD, la mejor configuración de la búsqueda del notebook 05; `HistGradientBoostingRegressor` para EAD y LGD.

## Decisiones técnicas relevantes

### Un único preprocesador scikit-learn ([DEC-006](decisions.md))

**Problema:** los encoders se ajustaban uno por uno, no se guardaban y las categorías raras se reagrupaban con `replace` de pandas; validación y producción no podían repetir la preparación. **Elección:** un `ColumnTransformer` guardado con joblib que agrupa categorías con menos de 200 casos, codifica las desconocidas sin fallar y aplica Yeo-Johnson a las numéricas asimétricas (la asimetría de `ingresos` bajó de 2,19 a 0,13). `01_reentrenamiento.py` usa la misma definición, así que las métricas de los notebooks describen el artefacto desplegado. One Hot con `drop='first'`: con k columnas una se deduce de las demás, y la regresión logística no tolera esa redundancia.

### Preprocesador ajustado dentro de la validación cruzada ([DEC-008](decisions.md))

**Problema:** si el escalado se ajusta con todas las filas antes de partir, el test influye en la preparación y la métrica sale optimista. **Elección:** el notebook 05 clona el preprocesador sin ajustar y lo mete en el `Pipeline`; cada fold lo ajusta solo con su parte de entrenamiento. Partición estratificada, `random_state=42`.

### Métricas de ordenamiento y de calibración ([DEC-008](decisions.md), [DEC-009](decisions.md))

**Problema:** el AUC dice si el modelo ordena, no si la probabilidad es correcta, y la pérdida esperada multiplica la probabilidad. **Elección:** AUC/Gini/KS para ordenamiento; Brier y curva de calibración para exactitud, sobre el test interno y sobre `validacion.pkl`. `03_validacion_externa.py` aplica los mismos filtros de filas que el entrenamiento.

### Sector de empleo en el contrato de la API ([DEC-010](decisions.md))

**Problema:** de las cuatro variables con señal que quedaban fuera, solo `sector_empleo` mejora el modelo de forma apreciable (+0,0036 de AUC; las otras tres, entre +0,0001 y +0,0007). **Elección:** la API recibe el título de empleo crudo (`empleo`, obligatorio pero admite `null`) y el pipeline lo convierte con `04_scripts/sector_empleo.py`, la misma regla de la fase de calidad. El código viaja embebido en el artefacto (`cloudpickle.register_pickle_by_value`), así que la API no depende de `04_scripts`. Un cliente que siga enviando los 14 campos viejos recibe un 422 explícito en lugar de quedar clasificado en silencio como `desconocido`, el sector de mayor riesgo.

### EAD y LGD comparados con un baseline ([DEC-011](decisions.md))

**Problema:** los notebooks 06 y 07 reentrenaban el modelo final con hiperparámetros escritos a mano distintos del ganador de la búsqueda, sin semilla y sin referencia. **Elección:** `best_estimator_`, `random_state=42`, preprocesador clonado dentro del `Pipeline` y comparación del MAE contra `DummyRegressor(strategy='median')`. Esa comparación reveló que el modelo de LGD casi no aprende.

### Sin rebalanceo de clases ([DEC-007](decisions.md))

**Problema:** la tasa de impago es 19,9% (16.568 casos), sin escasez de positivos. **Elección:** no remuestrear, porque desplazaría la PD hacia arriba y obligaría a recalibrarla antes de calcular la pérdida esperada.

### Umbral demostrativo ([DEC-004](decisions.md))

El umbral de pérdida esperada relativa `≤ 0.05` es una referencia visual no calibrada, no una política de riesgo ni una aprobación.

## Resultados principales

| Área analizada | Hallazgo | Interpretación |
|---|---|---|
| Ordenamiento PD | AUC 0,708 · Gini 0,42 · KS 0,31 en validación externa | Estable entre CV, test y validación externa; separa claramente mejor que el azar. |
| Calibración PD | Brier 0,144 frente a 0,158 sin modelo; ±3,0 p.p. por decil | La probabilidad se puede usar directamente en `PD × EAD × LGD`. |
| Captura de impagos | 20% de mayor PD → 39% de los impagos | Priorizar la revisión por PD casi duplica la eficiencia frente al azar. |
| EAD | MAE 0,159 vs 0,177 de la mediana (validación interna) | Mejora 9,8%; las predicciones van de 0,55 a 0,95 y sobrestiman los EAD bajos. |
| LGD | MAE 0,087 vs 0,088 de la mediana (validación interna) | Mejora 0,6%: con las variables actuales no distingue préstamos. |
| Sector de empleo | +0,0036 de AUC en CV; AUC externo 0,705 → 0,708 | Se incorporó al contrato de la API como título de empleo crudo. |
| Variables sin información | `num_meses_desde_ult_retraso`, `num_cancelaciones_12meses`: AUC ≈ 0,50 | Se excluyeron. |
| Variables con señal redundante | `num_hipotecas`, `porc_tarjetas_75p`, `tiene_descripcion`: AUC ≈ 0,55 solas, +0,0001 a +0,0007 en el modelo | Su información ya está en otras variables; no justifican ampliar el contrato. |

## Estructura del proyecto

```text
AA_scoring-crediticio/
├── 01_Documentos/
│   └── Diseño_Transformaciones.md        # Matriz de transformaciones, variable por variable
├── 02_datos/                             # No versionado
│   ├── 01_Originales/prestamos.csv       # Datos fuente (200.000 préstamos)
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
│   ├── 05_Modelizacion Clasificacion PD.ipynb  # Modelo PD, métricas de crédito, validación externa
│   ├── 06_Modelizacion Regresion EAD.ipynb     # Modelo EAD
│   ├── 07_Modelizacion Regresion LGD.ipynb     # Modelo LGD
│   └── 08_Preproduccion.ipynb            # Flujo consolidado
├── 06_resultados/Validacion/
│   ├── informe_validacion_modelos.md     # Informe de validación del artefacto
│   └── metricas_validacion_externa.json  # Métricas externas de PD, EAD y LGD
├── 07_despliegue/
│   ├── 01_reentrenamiento.py             # Entrena y genera el artefacto reproducible
│   ├── 03_validacion_externa.py          # Evalúa el artefacto sobre validacion.pkl
│   ├── api/
│   │   ├── main.py                       # FastAPI: /health, /predict, /docs
│   │   ├── scoring.py                    # Motor de scoring
│   │   ├── schemas.py                    # Contrato de entrada/salida
│   │   ├── artefacto_pipeline.pkl        # Pipelines PD, EAD y LGD (único .pkl versionado)
│   │   ├── requirements.txt              # Dependencias de la API
│   │   └── RENDER_DEPLOY.md              # Guía de despliegue de la API
│   └── app/
│       ├── app.py                        # Dashboard Streamlit
│       ├── app_core.py                   # Lógica del cliente y presentación
│       ├── tests/                        # Tests del cliente y de la integración HTTP
│       ├── requirements.txt              # Dependencias del dashboard
│       └── README.md                     # Instrucciones específicas de la app
├── odd/tasks/                            # Seguimiento de tareas de cada mejora
├── decisions.md                          # Decisiones técnicas (DEC-001 a DEC-011)
└── README.md                             # Este documento
```

`03_notebooks/` es una versión previa de los notebooks 01 y 08; el flujo vigente está en `Notebooks/`.

## Cómo reproducir

**Requisitos:** Python 3.13 o compatible con las versiones fijadas en los `requirements.txt` (pandas 3.0.5, scikit-learn 1.9.0, FastAPI 0.141.1, Streamlit 1.62.0). `02_datos/` y los `.pkl`/`.joblib` no se versionan; hay que colocar `prestamos.csv` en `02_datos/01_Originales/`.

**Entorno (Windows PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r 07_despliegue\api\requirements.txt
python -m pip install -r 07_despliegue\app\requirements.txt
python -m pip install jupyter matplotlib seaborn python-dotenv   # solo notebooks
```

En Linux/macOS: `source .venv/bin/activate` y `/` en lugar de `\`.

**Pasos:**

1. Ejecutá los notebooks en orden, desde `Notebooks/` (las rutas son relativas y los nombres llevan espacios). El 04 genera `preprocesador.joblib` y los tablones que usan 05–07; el 06 y el 07 tardan unos 10 minutos cada uno.

   ```powershell
   cd Notebooks
   foreach ($n in "01_ImportacionDatos","02_Calidad de Datos","03_EDA","04_Transformacion de datos","05_Modelizacion Clasificacion PD","06_Modelizacion Regresion EAD","07_Modelizacion Regresion LGD") {
       ..\.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace "$n.ipynb"
   }
   cd ..
   ```

2. (Opcional) Regenerá el artefacto: `python 07_despliegue\01_reentrenamiento.py`.
3. Iniciá la API:

   ```powershell
   cd 07_despliegue\api
   ..\..\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --app-dir ..
   ```

   Endpoints: `GET /health`, `POST /predict` (lista de registros crudos → `score_pd`, `score_ead`, `score_lgd`, `perdida_esperada_relativa`) y `GET /docs`.
4. En otra terminal, iniciá el dashboard desde la raíz: `.venv\Scripts\python.exe -m streamlit run 07_despliegue\app\app.py`. Usa `http://127.0.0.1:8000` por defecto; para una API desplegada, definí `API_BASE_URL`.

## Limitaciones y próximos pasos

- **LGD sin poder predictivo.** Mejora 0,6% sobre la mediana. Falta probar otras variables o un enfoque en dos etapas (clasificar si la LGD es 1 y luego regresión para el resto).
- **Producción desalineada en EAD y LGD.** `01_reentrenamiento.py` y `08_Preproduccion.ipynb` no incorporan las correcciones de [DEC-011](decisions.md).
- **Sin umbral de decisión.** Falta cuantificar costo de falsos positivos y beneficio de verdaderos positivos para elegir un umbral que maximice el valor esperado.
- **Colinealidad sin resolver.** `rating`/`tipo_interes` (0,95) y `principal`/`imp_cuota` (0,97) siguen juntas en el modelo; falta la fase de selección de variables.
- **Clientes repetidos entre entrenamiento y validación.** Ningún préstamo se repite, pero 41.895 `id_cliente` de validación aparecen también en entrenamiento; si un cliente tiene varios préstamos, la validación puede ser algo optimista.
- **Clasificación de empleo por palabras clave.** El 18,5% de los clientes de entrenamiento queda en `otros`. El dashboard envía siempre "Office Manager" (`administrativo`, 19,6% de impago, la tasa más cercana al promedio) como campo oculto.
- **Uso responsable.** El dashboard es demostrativo: no es aprobación, denegación, oferta ni *pricing*. EAD y LGD son proporciones condicionadas al impago, no montos absolutos. No subas credenciales ni datos sensibles al repositorio.
