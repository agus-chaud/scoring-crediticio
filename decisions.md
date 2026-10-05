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
