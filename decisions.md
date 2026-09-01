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