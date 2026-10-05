# Diseño de transformaciones — CONGELADO

- **Proyecto:** Scoring crediticio (PD, EAD, LGD)
- **Fecha:** 2026-10-05
- **Objetivo:** clasificación binaria — probabilidad de impago (PD); los mismos features alimentan EAD y LGD (regresión).
- **Target:** `target_pd` (EAD: `target_ead`, LGD: `target_lgd`, solo sobre defaults)
- **Modelos priorizados:** regresión logística (lineal) para PD; árboles para EAD y LGD.
- **Implementación:** `05_modelos/preprocesador.joblib` (`ColumnTransformer`), generado por `Notebooks/04_Transformacion de datos.ipynb`.

| Variable | Tipo_Original | Transformación_1 | Tipo_Resultado_1 | Transformación_2 | Tipo_Resultado_2 | Escalado_Final | Es_Final | Incluir_DF | Nombre_Col_Final | Justificación |
|---|---|---|---|---|---|---|---|---|---|---|
| ingresos_verificados | cat_nominal | OHE(drop='first', min_frequency=200) | binaria | — | — | NO | SÍ | SÍ | `ohe__ingresos_verificados_*` | 3 niveles; `drop='first'` evita multicolinealidad perfecta en la logística. |
| vivienda | cat_nominal | OHE(drop='first', min_frequency=200) | binaria | — | — | NO | SÍ | SÍ | `ohe__vivienda_*` | ANY/NONE/OTHER (28 filas) van a `infrequent_sklearn` dentro del preprocesador. |
| finalidad | cat_nominal | OHE(drop='first', min_frequency=200) | binaria | — | — | NO | SÍ | SÍ | `ohe__finalidad_*` | wedding/renewable_energy/educational (223 filas) van a `infrequent_sklearn`. |
| num_cuotas | binaria | OHE(drop='first') | binaria | — | — | NO | SÍ | SÍ | `ohe__num_cuotas_ 60 months` | 2 niveles: una sola columna (antes eran 2 con correlación 1,0). |
| antigüedad_empleo | cat_ordinal | OrdinalEncoder (orden explícito, unknown=-1) | num_discreta | StandardScaler | num_continua | StandardScaler | SÍ | SÍ | `oe__antigüedad_empleo` | `desconocido` es el nivel más bajo; las ordinales se escalan. |
| rating | cat_ordinal | OrdinalEncoder (A→G, unknown=-1) | num_discreta | StandardScaler | num_continua | StandardScaler | SÍ | SÍ | `oe__rating` | `unknown_value=-1` evita el atípico artificial que generaba 12. |
| ingresos | num_continua | Yeo-Johnson (standardize) | num_continua | — | — | Incluido en Yeo-Johnson | SÍ | SÍ | `yj__ingresos` | Asimetría 2,19 → 0,13. |
| dti | num_continua | Yeo-Johnson (standardize) | num_continua | — | — | Incluido en Yeo-Johnson | SÍ | SÍ | `yj__dti` | Asimetría 0,87 → 0,02. |
| num_lineas_credito | num_discreta | Yeo-Johnson (standardize) | num_continua | — | — | Incluido en Yeo-Johnson | SÍ | SÍ | `yj__num_lineas_credito` | Asimetría 1,26 → 0,00. |
| principal | num_continua | Yeo-Johnson (standardize) | num_continua | — | — | Incluido en Yeo-Johnson | SÍ | SÍ | `yj__principal` | Asimetría 0,79 → −0,04. |
| imp_cuota | num_continua | Yeo-Johnson (standardize) | num_continua | — | — | Incluido en Yeo-Johnson | SÍ | SÍ | `yj__imp_cuota` | Asimetría 1,01 → −0,02. |
| porc_uso_revolving | num_continua | — | — | — | — | StandardScaler | SÍ | SÍ | `ss__porc_uso_revolving` | Asimetría −0,08: solo escalado. |
| tipo_interes | num_continua | — | — | — | — | StandardScaler | SÍ | SÍ | `ss__tipo_interes` | Asimetría 0,70 (< 0,75): solo escalado. |
| num_derogatorios | num_discreta | Binarizer(threshold=0) | binaria | — | — | NO | SÍ | SÍ | `bin__num_derogatorios` | 1 = al menos un derogatorio (14.147 filas). |
| num_meses_desde_ult_retraso | num_continua | — | — | — | — | — | — | NO | — | AUC univariante 0,504; tasa de impago 19,8% / 20,5% / 19,7% por tramo de cuantiles. |
| num_cancelaciones_12meses | num_discreta | — | — | — | — | — | — | NO | — | AUC univariante 0,501. |
| num_hipotecas | num_discreta | — | — | — | — | — | — | NO | — | AUC 0,547: tiene señal. Fuera para mantener el contrato de 14 campos de la API; candidata para selección de variables. |
| porc_tarjetas_75p | num_continua | — | — | — | — | — | — | NO | — | AUC 0,552: tiene señal. Igual que `num_hipotecas`. |
| tiene_descripcion | binaria | — | — | — | — | — | — | NO | — | Impago 15,7% con descripción vs 20,3% sin ella. Candidata. |
| sector_empleo | cat_nominal | — | — | — | — | — | — | NO | — | Impago 26,6% en `desconocido` vs 14–25% en el resto. Candidata. |
| imp_amortizado, imp_recuperado, estado | — | — | — | — | — | — | — | NO | — | Se conocen después del desenlace del préstamo: fuga de información. Solo se usan para construir las targets. |
| id_cliente | — | — | — | — | — | — | — | NO (índice) | — | Identificador. |
| target_pd / target_ead / target_lgd | target | — | — | — | — | NO | SÍ | SÍ | igual | La target se incluye sin transformar. |

## Decisiones tomadas

- [DEC-006](../decisions.md#dec-006-preparación-de-variables-en-un-único-preprocesador-scikit-learn): estrategia de transformación (este documento).
- [DEC-007](../decisions.md#dec-007-sin-rebalanceo-de-clases-para-pd): sin rebalanceo de clases.
- [DEC-008](../decisions.md#dec-008-modelización-pd-con-preprocesador-dentro-de-la-validación-cruzada-y-validación-externa): el preprocesador se clona y ajusta dentro de la validación cruzada del modelo PD.
- Se mantiene el contrato de 14 variables crudas de la API ([DEC-001](../decisions.md)).

## Salidas

| Archivo | Contenido |
|---|---|
| `05_modelos/preprocesador.joblib` | `ColumnTransformer` ajustado sobre las 83.250 filas de entrenamiento. |
| `02_datos/03_Entrenamiento/df_tablon_pd_sin_transformar.pkl` | 14 variables crudas + `target_pd`, índice `id_cliente`. Entrada del notebook 05. |
| `02_datos/03_Entrenamiento/df_tablon_pd.pkl` | 27 columnas transformadas + `target_pd` (83.250 filas). |
| `02_datos/03_Entrenamiento/df_tablon_ead.pkl`, `df_tablon_lgd.pkl` | 27 columnas transformadas + target, solo defaults (16.568 filas). |

## Validaciones realizadas

Filas preservadas, target alineada por `id_cliente`, sin variables originales ni intermedias en la salida, sin nulos,
sin columnas duplicadas, sin multicolinealidad perfecta entre binarias y una fila con categorías inventadas
transformada sin error. El preprocesador recargado desde joblib reproduce el tablón exactamente.

## Riesgos identificados

- **Colinealidad:** en el tablón transformado, `oe__rating` vs `ss__tipo_interes` tienen correlación 0,95 y `yj__principal` vs `yj__imp_cuota`, 0,97. Con Ridge, los coeficientes de `rating` y `tipo_interes` salen los dos positivos (con el diseño anterior, `tipo_interes` salía negativo), pero aportan casi la misma información: la fase de selección de variables debería quedarse con una de cada par.
- **Valores extremos bajos en `yj__ingresos`** (mínimo −9,3, por ingresos iguales a 0): son pocos casos, conviene revisarlos en calidad.
- **Divergencia con producción:** `07_despliegue/01_reentrenamiento.py` todavía usa MinMax y reagrupa con pandas. Si se adopta este diseño, hay que alinearlo.
