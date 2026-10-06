# Technical decisions

## DEC-001 — Reconstructed production preprocessing and three risk models

- **Area:** deployment
- **Decision:** Rebuild the raw-input preprocessing inside each released sklearn pipeline and train three independent models: PD, EAD and LGD.
- **Alternative discarded:** Reuse the already transformed tables as the public API contract.
- **Why discarded:** Those tables contain engineered numeric columns only; making them the API input would expose an unusable contract and would not reproduce the raw-loan preparation.
- **Conclusion:** The API receives 14 raw loan fields. Each model owns the same deterministic preparation chain, fitted only on training data. EAD and LGD are trained on defaulted loans, as defined by the original notebooks.

## DEC-002 — Production-safe rebuild differs from the historical notebook

- **Area:** deployment
- **Decision:** Use `random_state=42`, fit preparation inside model pipelines, and apply `OneHotEncoder(drop='first')`.
- **Alternative discarded:** Copy the historical notebook behavior verbatim.
- **Why discarded:** The notebook did not fix its split seed and fitted encoders/scalers before validation, which can make validation look better than it really is.
- **Conclusion:** The released artifact favors repeatable, safer training over byte-for-byte reproduction of the historic prepared tables.

## DEC-003: Cliente Streamlit fino y contrato crudo

**Área:** despliegue | **Fase:** interfaz Streamlit | **Fecha:** 2026-09-01 | **Estado:** Vigente

**Decisión:** La nueva interfaz en `07_despliegue/app` consume exclusivamente `07_despliegue/api` mediante `POST /predict` y transmite el registro como lista sin transformar categorías ni escalas.

**Alternativa descartada:** Reutilizar las aplicaciones heredadas, cargar el artefacto de ML o normalizar `tipo_interes` y `num_cuotas` en el cliente.

**Por qué la descartamos:** Duplicaría el backend de scoring, introduciría deriva semántica y podría modificar una decisión demostrativa sin una definición explícita aguas arriba.

**Conclusión:** Siempre usar la API canónica como único backend de scoring; nunca cargar artefactos ni corregir silenciosamente valores crudos desde la interfaz.

## DEC-004: Umbral y sensibilidad sólo demostrativos

**Área:** despliegue | **Fase:** interfaz Streamlit | **Fecha:** 2026-09-01 | **Estado:** Vigente

**Decisión:** La app muestra un umbral configurable de pérdida esperada relativa `<= 0.05` y escenarios ceteris-paribus que sólo modifican `principal` y `num_cuotas`, sin aumentar el principal y con un máximo de seis resultados calificables.

**Alternativa descartada:** Presentar aprobaciones, ofertas o recomendaciones de monto/plazo a partir del resultado del modelo.

**Por qué la descartamos:** El umbral no fue calibrado como política de riesgo y mantener la cuota fija vuelve incompleta cualquier lectura económica del escenario.

**Conclusión:** Si un resultado no tiene política calibrada y validada, etiquetarlo como demostrativo; nunca convertir sensibilidad de modelo en oferta, aprobación o recomendación.

## DEC-005: Rediseño del tablero — ocho filtros visibles, resto oculto y fijo

**Área:** despliegue | **Fase:** interfaz Streamlit | **Fecha:** 2026-09-01 | **Estado:** Vigente

**Decisión:** La interfaz pasa a una composición de tablero analítico: panel de filtros angosto a la izquierda (25%) y área de resultados a la derecha. Sólo se muestran ocho filtros editables: `principal`, `num_cuotas`, `tipo_interes`, `imp_cuota`, `ingresos`, `dti`, `porc_uso_revolving` y `rating`. Los otros seis campos obligatorios del contrato (`ingresos_verificados`, `vivienda`, `finalidad`, `antigüedad_empleo`, `num_lineas_credito`, `num_derogatorios`) quedan en `hidden_fields` del `design_spec.json` con su valor exacto de `test_payload.json` y se envían en cada request vía `compose_payload_values`.

**Actualización (DEC-012):** `sector_empleo` se sumó después como noveno filtro visible (selectbox de 14 sectores); los campos ocultos siguen siendo seis y `empleo` ya no forma parte del contrato.

**Por qué esos ocho:** son los que explican de forma directa el riesgo y las condiciones del préstamo (monto, plazo, tasa, cuota, capacidad de pago vía ingresos y DTI, comportamiento revolving y rating interno). El resto son atributos de contexto que, para una visualización demostrativa, aportan poco valor de exploración y recargan el panel.

**Cómo se preservan los ocultos:** `compose_payload_values(visible, hidden)` parte de una copia de `hidden` y la pisa con `visible`; nunca muta los argumentos. `build_payload` sigue validando los 14 `REQUIRED_FIELDS`, así que si alguien quita un campo oculto del spec el envío falla de forma explícita en vez de mandar un payload incompleto. Tests: `test_hidden_fields_reach_the_payload_with_fixture_values`, `test_eight_visible_fields_are_sent_and_win_over_hidden`, `test_compose_payload_values_does_not_mutate_inputs`.

**Alternativa descartada:** dejar los 14 campos como inputs, o normalizar/derivar los ocultos en el cliente. Se descartó porque recarga la interfaz y porque tocar los valores crudos rompería DEC-003.

**Visualización principal:** tarjeta azul con `perdida_esperada_relativa` en porcentaje + barra horizontal de impacto + indicador tipo gauge, todos sobre una escala demostrativa 0–25% con bandas en 5% y 10%. Se eligió gauge + barra porque comunican una sola métrica acotada contra una referencia de un vistazo; PD, EAD y LGD quedan como métricas secundarias. Cada estado trae etiqueta de texto (`risk_band` → "Riesgo bajo/medio/alto", más una línea concreta del tipo "La pérdida estimada superó el 10% de referencia"); las dos visualizaciones Plotly incluyen texto equivalente, no dependen del color. Las etiquetas son una lectura de riesgo demostrativa, no una decisión de crédito (DEC-004). Se descartó "Dentro/Fuera del umbral demostrativo" por poco claro. Se agregó `plotly==7.0.0` (versión del entorno confirmado).

**Se eliminaron** los dos avisos técnicos de la interfaz (`Visualización demostrativa conectada a la API canónica.` y el aviso de valores crudos / no normalización), sin reemplazarlos por otros avisos técnicos. La app sigue enviando los valores crudos sin transformarlos.

**Conclusión:** el panel muestra sólo lo que aporta a explicar riesgo; los campos de contrato que no se muestran viven en `hidden_fields` y se inyectan siempre; la lectura principal es una métrica única contra umbral, con texto siempre presente.

## DEC-006: Preparación de variables en un único preprocesador scikit-learn

**Área:** feature-engineering | **Fase:** A_04 Transformación | **Fecha:** 2026-10-05 | **Estado:** Vigente

**Decisión:** `Notebooks/04_Transformacion de datos.ipynb` concentra toda la preparación de las 14 variables crudas en un `ColumnTransformer` guardado en `05_modelos/preprocesador.joblib`: `OneHotEncoder(drop='first', min_frequency=200, handle_unknown='infrequent_if_exist')`, `OrdinalEncoder` con orden explícito y `unknown_value=-1` seguido de `StandardScaler`, Yeo-Johnson para las numéricas con asimetría mayor a 0,75 (`ingresos`, `dti`, `num_lineas_credito`, `principal`, `imp_cuota`), `StandardScaler` para `porc_uso_revolving` y `tipo_interes`, y `Binarizer` para `num_derogatorios`. El detalle variable por variable está en `01_Documentos/Diseño_Transformaciones.md`.

**Alternativa descartada:** Encoders sueltos ajustados uno por uno, reagrupación de categorías raras con `replace` de pandas, `MinMaxScaler` para todas las numéricas, k dummies por variable y `unknown_value=12`.

**Por qué la descartamos:** Los pasos sueltos no se guardaban, así que la validación y producción no podían repetirlos. El `replace` quedaba fuera de cualquier pipeline. MinMax no corrige la asimetría (`ingresos`: 2,19), y con eso el 75% de los clientes quedaba por debajo de 0,225. Con k dummies, `num_cuotas` generaba dos columnas con correlación 1,0. Un `rating` desconocido codificado como 12 terminaba en 2,0 después del escalado, un valor atípico artificial.

**Conclusión:** Toda transformación de variables predictoras vive en un objeto scikit-learn persistido y se ajusta solo con datos de entrenamiento. Toda variable excluida se documenta con evidencia (AUC univariante o tasa de impago por nivel). Esta decisión actualiza la premisa de DEC-002 sobre el notebook. `07_despliegue/01_reentrenamiento.py` replica este preprocesador y la configuración de PD de DEC-008 desde el artefacto 1.2.0.

## DEC-007: Sin rebalanceo de clases para PD

**Área:** modelado | **Fase:** A_06 Balanceo | **Fecha:** 2026-10-05 | **Estado:** Vigente

**Decisión:** El modelo de PD se entrena con la proporción real de clases, sin remuestreo ni `class_weight`.

**Alternativa descartada:** `class_weight='balanced'`, RandomUnderSampler, RandomOverSampler y SMOTENC-Tomek.

**Por qué la descartamos:** La tasa de impago es 19,9% (16.568 positivos sobre 83.250, unos 570 por variable): no hay escasez de casos positivos. Además, la PD se multiplica por EAD y LGD para calcular la pérdida esperada. El remuestreo desplaza las probabilidades predichas hacia arriba y obliga a recalibrarlas. Sin rebalanceo, la PD media predicha en validación externa es 19,8% frente a una tasa real de 19,7%.

**Conclusión:** Si la probabilidad se usa como magnitud (pérdida esperada, pricing) y la clase minoritaria supera el 15% con miles de casos, no rebalancear. Para ganar recall, mover el umbral de decisión en lugar de alterar los datos.

## DEC-008: Modelización PD con preprocesador dentro de la validación cruzada y validación externa

**Área:** modelado | **Fase:** A_05 Modelización PD | **Fecha:** 2026-10-05 | **Estado:** Vigente

**Decisión:** `Notebooks/05_Modelizacion Clasificacion PD.ipynb` lee el tablón sin transformar y clona `preprocesador.joblib` dentro del `Pipeline`. Usa partición estratificada con `random_state=42`, `StratifiedKFold(5)` y una grilla `C = logspace(-3, 2, 11)` × `l1_ratio ∈ {0; 0,5; 1}`. Toma `best_estimator_` y evalúa sobre el test interno y sobre `02_datos/02_Validacion/validacion.pkl`, con AUC, Gini, KS, Brier y curva de calibración.

**Alternativa descartada:** Entrenar sobre el tablón ya escalado con las 83.250 filas, partir sin semilla ni estratificación, usar una grilla de `C` entre 0,01 y 1 y reportar solo el AUC.

**Por qué la descartamos:** Escalar antes de partir deja que el test influya en la preparación (fuga de información). Sin semilla, el resultado no se puede reproducir. En la grilla anterior el mejor `C` caía en el borde. El AUC solo mide el ordenamiento, no si la probabilidad es correcta, y eso es lo que necesita la pérdida esperada.

**Conclusión:** Resultado: Ridge (`l1_ratio=0`) con `C=0,01`. AUC 0,703 en validación cruzada, 0,705 en test interno y 0,705 en validación externa (35.592 préstamos con desenlace); Gini 0,41; KS 0,31; Brier 0,145 frente a 0,158 sin modelo; calibración dentro de ±2,6 p.p. por decil. Todo modelo nuevo se evalúa con métricas de ordenamiento y de calibración sobre `validacion.pkl`.


## DEC-009: Validación externa del despliegue con los mismos filtros que el entrenamiento

**Área:** despliegue | **Fase:** validación externa | **Fecha:** 2026-10-05 | **Estado:** Vigente

**Decisión:** `07_despliegue/03_validacion_externa.py` excluye de `validacion.pkl` los préstamos sin desenlace (`Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`) y los `dti = 999`, igual que `01_reentrenamiento.py` y la fase de calidad.

**Alternativa descartada:** Evaluar todas las filas con ingresos <= 400.000, como hacía el script.

**Por qué la descartamos:** Un préstamo vigente todavía no pagó ni incumplió, así que contarlo como buen pagador inventa la etiqueta. Con esos préstamos incluidos se evaluaban 59.833 filas con una tasa de impago "real" de 11,7%, frente a una PD media predicha de 19,9%: el modelo parecía descalibrado y su AUC bajaba a 0,686. Con los mismos filtros quedan 35.592 préstamos, tasa real 19,7%, PD media 19,9% y AUC 0,706.

**Conclusión:** Validación y entrenamiento comparten siempre la definición de población y de target. Si cambia un filtro de filas, cambia en los dos scripts.


## DEC-010: Sector de empleo en el modelo y en el contrato de la API

**Área:** modelado | **Fase:** A_05 Modelización PD / despliegue | **Fecha:** 2026-10-05 | **Estado:** Vigente (parcialmente reemplazada por DEC-012)

> Nota: DEC-012 reemplaza la parte del contrato de la API: el campo `empleo` y el embebido de `sector_empleo.py` en el artefacto pasaron a un campo `sector_empleo` ya resuelto (artefacto 2.0.0). La incorporación de `sector_empleo` al modelo sigue vigente.

**Decisión:** `sector_empleo` entra al preprocesador como One Hot. La API agrega el campo `empleo` (título de empleo crudo, obligatorio, admite `null`) y el pipeline lo convierte en sector con `04_scripts/sector_empleo.py`, copia exacta de la regla de `02_Calidad de Datos` (verificada sobre las 83.250 filas de entrenamiento). El módulo se embebe en el artefacto 1.3.0 con `cloudpickle.register_pickle_by_value`. La app envía "Office Manager" como campo oculto. El contrato pasa de los 14 campos de DEC-001 y DEC-005 a 15.

**Alternativa descartada:** Agregar también `num_hipotecas`, `porc_tarjetas_75p` y `tiene_descripcion` (18 campos), o no cambiar el contrato.

**Por qué la descartamos:** En validación cruzada (Ridge, 5 folds, ruido ±0,005), las tres variables aportan +0,0005, +0,0001 y +0,0007 de AUC: no compensan el costo de ampliar el contrato. `sector_empleo` aporta +0,0036 y captura un grupo de riesgo claro (sin título de empleo: 26,6% de impago). Que el campo sea obligatorio evita que un cliente viejo quede clasificado en silencio como `desconocido`: recibe un 422.

**Conclusión:** Resultado: mejor configuración Ridge con `C≈0,316`. AUC 0,707 en validación cruzada y 0,708 en `validacion.pkl` en el notebook; 0,709 en la validación del artefacto. Una variable entra al contrato de la API solo si su aporte dentro del modelo supera el ruido entre folds, no solo por su señal univariante. Toda regla de derivación usada en producción vive en un único módulo compartido por notebooks y reentrenamiento.


## DEC-011: Modelización EAD y LGD con el ganador del grid, baseline y preprocesador dentro de la validación cruzada

**Área:** modelado | **Fase:** A_06/A_07 Modelización EAD y LGD | **Fecha:** 2026-10-05 | **Estado:** Vigente

**Decisión:** `06_Modelizacion Regresion EAD.ipynb` y `07_Modelizacion Regresion LGD.ipynb` leen tablones sin transformar (`df_tablon_ead_sin_transformar.pkl`, `df_tablon_lgd_sin_transformar.pkl`, exportados por el notebook 04) y clonan `preprocesador.joblib` dentro del `Pipeline`. Partición con `random_state=42` y `KFold(5, shuffle=True)`. Ridge y Lasso llevan `StandardScaler` y `alpha = logspace(-3, 2, 11)`; `HistGradientBoostingRegressor` usa `loss='absolute_error'` sin `scoring` MAPE. El modelo final es `best_estimator_` y se compara con `DummyRegressor(strategy='median')`. Los residuos se grafican contra el valor predicho.

**Alternativa descartada:** Tablones ya transformados, partición sin semilla, `cv=3` sin mezclar, `alpha` lineal entre 0,1 y 1, HGB con parada temprana por MAPE y un modelo final reentrenado con hiperparámetros escritos a mano distintos del ganador.

**Por qué la descartamos:** El modelo evaluado no era el ganador de la búsqueda, y sin semilla cada ejecución daba otro resultado. El preprocesador ajustado con todo el train filtraba estadísticos a la validación interna. El MAPE explota con EAD cercanos a 0 y no coincidía con el MAE de la búsqueda. Sin baseline, un MAE de 0,09 en LGD parecía bueno. Graficar residuos contra el valor real muestra siempre una pendiente positiva, aunque el modelo no tenga sesgo.

**Conclusión:** Resultado: EAD, HGB (lr 0,05, depth 10, 200 iteraciones, l2 0,25), MAE 0,154 en CV y 0,159 en validación interna frente a 0,177 de la mediana (−9,8%). LGD, HGB (lr 0,01, depth 5, 200 iteraciones, l2 0,75), MAE 0,089 en CV y 0,087 frente a 0,088 de la mediana (−0,6%): el modelo de LGD prácticamente no aprende, predice entre 0,88 y 0,92 para todos. Todo modelo de regresión se reporta junto a un baseline trivial. El artefacto de producción (`01_reentrenamiento.py`) todavía no incorpora estos cambios.


## DEC-012: La API recibe `sector_empleo` en lugar del título de empleo crudo

**Área:** despliegue | **Fase:** contrato de la API / dashboard | **Fecha:** 2026-10-06 | **Estado:** Vigente

**Decisión:** el contrato de `POST /predict` reemplaza `empleo` por `sector_empleo`, obligatorio y restringido a 14 valores: `legal`, `salud`, `educacion`, `finanzas`, `seguridad_publica`, `tecnologia`, `transporte`, `hosteleria`, `oficios`, `ventas`, `administrativo`, `direccion`, `otros`, `desconocido`. El contrato sigue en 15 campos. Los campos extra se rechazan con 422, y un `sector_empleo` ausente, `null` o fuera de la lista también. `clean_raw_input` valida el sector contra `SECTORES_VALIDOS` en lugar de derivarlo. `04_scripts/sector_empleo.py` solo se usa para derivar el sector desde `empleo` en los datos de entrenamiento y validación. Artefacto 2.0.0 (antes 1.3.0). El dashboard muestra un selectbox "Sector de empleo" (valor inicial `administrativo`) y ya no envía "Office Manager".

**Por qué:** DEC-010 dejó `empleo` como campo oculto fijo en "Office Manager", que la regla de palabras clave clasifica como `administrativo`. Todos los clientes puntuados desde el dashboard quedaban como administrativos (19,6% de impago), así que el sector no cambiaba el resultado aunque el modelo lo usa (+0,0036 de AUC en CV).

**Alternativas descartadas:**
- **A. Enviar desde la app un título representativo por sector sin cambiar la API.** Es frágil: depende de que cada título caiga en el sector esperado según las expresiones regulares de `sector_empleo.py`, y esa dependencia queda oculta en el cliente.
- **C. Selectbox con títulos de ejemplo.** Mantiene el mismo acoplamiento con las regex que A y agrega una lista de títulos que mantener.

**Consecuencias:**
- Cambio incompatible: un cliente que envíe `empleo` recibe 422.
- El artefacto 2.0.0 y el schema de la API se despliegan juntos; con el artefacto 1.3.0 y el schema nuevo, `/predict` responde 500.
- `null` ya no se transforma en `desconocido` (antes, `empleo: null` sí). Para ese caso el cliente debe enviar `desconocido` de forma explícita.
- Los clientes pasan a ser responsables de mapear un título a un sector. Un mapeo propio que difiera de `04_scripts/sector_empleo.py` cambia los scores. La regla por palabras clave deja el 18,5% de los títulos de entrenamiento en `otros`.

**Verificación:** el reentrenamiento da las mismas métricas que el artefacto 1.3.0 (PD ROC-AUC 0,7077 en train y 0,7087 en validación externa; EAD MAE 0,1606; LGD MAE 0,0903). `/predict` responde 200 con un sector válido y 422 con sector inválido, ausente o `null`, y con `empleo`. El selectbox de la app envía el sector elegido (19 tests de la app en verde).

**Conclusión:** el contrato recibe la variable que usa el modelo, no la entrada de la que se deriva. Cualquier regla de derivación que el cliente replique tiene que verificarse contra la regla de entrenamiento.


## DEC-013: Modelo de PD: regresión logística Ridge con splines en 4 variables, en lugar de EBM

**Área:** modelado | **Fase:** A_05 Modelización PD | **Fecha:** 2026-10-06 | **Estado:** Vigente

**Decisión:** usar para la PD la regresión logística Ridge con `SplineTransformer(degree=3, knots='quantile', include_bias=False, n_knots=8)` aplicado, dentro del `Pipeline` y después del preprocesador, a `yj__principal`, `yj__dti`, `yj__imp_cuota` y `ss__tipo_interes`; `C≈0,316`, `l1_ratio=0`. Producción (`07_despliegue/01_reentrenamiento.py`, función `make_pd_pipeline`) fija `n_knots=8` y `C=10**-0.5` sin repetir la búsqueda: artefacto 2.1.0 (antes 2.0.0), con el contrato de la API sin cambios. El notebook 05 incorpora la sección "Challenger: regresión logística con splines" y evalúa el modelo final con los gráficos de reporting; la logística lineal de DEC-008 queda como baseline (`modelo`, `pred_val_lineal`). No reemplazar la logística por un Explainable Boosting Machine (EBM).

**Por qué:** la logística lineal sobreestimaba la PD en los extremos de riesgo. En validación externa, el decil de menor riesgo tenía PD media 6,1% frente a una tasa real de 4,6%, y el de mayor riesgo 47,6% frente a 44,6%. Un EBM entrenado como challenger mostró que `tipo_interes` (R² lineal 0,76), `dti` (0,75), `imp_cuota` (0,52) y `principal` (0,40) tienen efectos no lineales; `rating`, `ingresos` y `num_lineas_credito` son casi lineales (R² 0,90 a 0,97). Las interacciones del EBM suman solo ~4% de la importancia total (0,071 de 1,69), así que la ganancia viene de las curvas por variable, no de interacciones.

**Resultados** (mismo split `random_state=42`, mismo preprocesador clonado dentro de `StratifiedKFold(5)`, mismos filtros de validación externa de DEC-009; 35.592 préstamos externos):

| Modelo | AUC CV | Gini test | Gini externo | Brier externo | Error de calibración por deciles | Impagos capturados en el 20% de mayor PD |
|---|---|---|---|---|---|---|
| Logística Ridge lineal (campeón) | 0,7074 | 0,415 | 0,416 | 0,1442 | 0,0111 | 39,4% |
| Logística + splines, 4 variables | 0,7098 | 0,418 | 0,423 | 0,1435 | 0,0036 | 39,8% |
| Logística + splines, 7 variables continuas | 0,7096 | 0,419 | 0,422 | 0,1435 | 0,0048 | 39,9% |
| EBM (`interactions=10`, `min_samples_leaf=50`) | 0,7106 | 0,422 | 0,429 | 0,1430 | 0,0052 | 40,2% |
| `HistGradientBoostingClassifier` sin ajustar (referencia) | n/d | n/d | 0,453 | 0,1412 | n/d | 41,5% |

- Las splines en 4 variables suman +0,006 de Gini externo sobre el campeón (IC95% bootstrap de 1.000 remuestreos: +0,004 a +0,009) y ganan en los 5 folds (+0,0024 de AUC medio). Cierran el 49% de la brecha de Gini con el EBM (+0,0126, IC95% +0,0095 a +0,0159). El resto de la brecha, 0,0065 (IC95% 0,0045 a 0,0085), también es estadísticamente distinto de cero.
- La calibración queda mejor que la del EBM: decil de menor riesgo 4,2% predicho frente a 4,1% real, decil de mayor riesgo 45,5% frente a 44,7%.
- Las curvas aprendidas coinciden con las del EBM. Contribución al log-odds en el tramo más bajo de `tipo_interes`: −0,822 (spline) y −0,815 (EBM); en `dti`: −0,133 y −0,132.
- Con 7 variables no mejora respecto de 4 (Gini externo 0,4222 contra 0,4225).
- En `principal`, la logística lineal asigna un efecto decreciente (+0,16 en montos bajos a −0,15 en altos), mientras que la spline y el EBM muestran que sube y luego baja. Hipótesis sin verificar: la colinealidad con `imp_cuota` (0,97) reparte el efecto de forma arbitraria en el modelo lineal.

**Alternativas descartadas:**
- **A. Reemplazar por EBM.** Gana +0,013 de Gini externo y calibra bien, pero exige la dependencia `interpret`, un artefacto de 4,3 MB, entrenar entre 25 y 45 s por fit (grilla de 20 fits: 753 s) y cambiar el reentrenamiento y el despliegue, para sumar solo +0,0065 de Gini sobre las splines.
- **B. Gradient boosting como modelo de PD.** Llega a +0,037 de Gini externo sobre la logística, la mayor ganancia medida, pero pierde la explicación por variable. Esa pérdida es una decisión de riesgo y regulación, no solo técnica, y queda abierta.
- **C. Splines en las 7 variables continuas.** No mejora el rendimiento ni la calibración (0,0048 contra 0,0036) y agrega columnas.

**Consecuencias:**
- La exploración con la grilla completa (`n_knots` {4, 6, 8} × `C = logspace(-3, 2, 11)`, 33 combinaciones) tardó 2.123 s con 4 variables y 1.306 s con 7, contra segundos para la logística lineal. El notebook 05 usa una grilla más corta alrededor del óptimo (`C = logspace(-2, 1, 7)`, 21 combinaciones): tarda 7 minutos y devuelve el mismo óptimo (`n_knots=8`, `C≈0,316`, AUC CV 0,7098). Producción fija esos valores en lugar de volver a buscar.
- No agrega dependencias: `SplineTransformer` es de scikit-learn. El artefacto sigue siendo una logística; cada variable con spline se explica con una curva en lugar de un coeficiente.
- `n_knots=8` es el borde de la grilla probada (4, 6, 8). La diferencia entre 4 y 8 nudos es de 0,0001 de AUC, pero no se probaron más.

**Límites de la evidencia:**
- Una sola partición train/test; el bootstrap mide solo la variabilidad de la muestra de validación, no la de reentrenar.
- El campeón se volvió a ajustar con los hiperparámetros ya reportados (Ridge, `C≈0,316`), no se volvió a buscar. La grilla del EBM fue chica (4 combinaciones) y el boosting de referencia usó valores por defecto.
- La calibración por deciles se calcula sobre los deciles de riesgo de cada modelo, de modo que cada fila agrupa clientes distintos.
- En el notebook 05, el boosting y el EBM son referencias con parámetros fijos: el EBM usa los valores de una grilla previa de 4 combinaciones, que no se repite, y requiere la librería opcional `interpret` (si falta, la celda lo avisa y el notebook sigue).
- El artefacto 2.1.0 se entrenó con `RATING_ORDER` de G a A, el mismo orden que usa el preprocesador del notebook 04. El script de reentrenamiento y el notebook 08 se alinearon con ese orden en un commit aparte. Con Ridge, invertir el orden de una variable ordinal solo cambia el signo de su coeficiente, así que no altera las predicciones.

**Verificación de la integración:** el notebook 05 ejecuta de punta a punta sin errores y reproduce el experimento: Gini externo 0,4225, error medio por decil 0,0036, diferencia de Gini frente a la lineal +0,0062 (IC95% +0,0037 a +0,0089, 1.000 remuestreos) y ganancia en los 5 folds. Reentrenamiento: PD ROC-AUC 0,7091 en el test interno; EAD (MAE 0,1606) y LGD (MAE 0,0903) no cambian. Validación externa del artefacto 2.1.0 sobre 35.592 préstamos: PD ROC-AUC 0,7119 (con 2.0.0, 0,7087); los informes de `06_resultados/Validacion/` se regeneran con ese valor. `/predict` responde 200 con un sector válido y 422 con uno inválido; los 19 tests de la app pasan.

**Conclusión:** la brecha entre la logística y el EBM era, en buena parte, falta de curvas en cuatro variables. Con splines la logística sube +0,006 de Gini, corrige los extremos de calibración y sigue siendo un modelo simple y explicable. Queda abierta la decisión de si el negocio justifica evaluar el boosting a cambio de la explicación por variable.
