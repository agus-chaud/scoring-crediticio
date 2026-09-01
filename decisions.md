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