# Feature: sector_empleo en el contrato de la API

## Objective
El usuario del dashboard elige su sector de empleo en un selectbox. La API recibe `sector_empleo` directamente (en lugar del título crudo `empleo`) y el dashboard deja de enviar siempre "Office Manager".

## Why
Hoy `empleo` va como campo oculto fijo ("Office Manager" -> `administrativo`), por lo que todos los clientes se puntúan como administrativos. Decisión del usuario: opción B (cambiar el contrato de la API).

## Scope
- Reentrenamiento: `clean_raw_input` deja de derivar el sector; `RAW_INPUT_COLUMNS` usa `sector_empleo`; el sector se deriva de `empleo` solo en datos de entrenamiento/validación.
- API: `schemas.py` con `sector_empleo` (Literal de 14 valores), payloads de ejemplo.
- Artefacto: reexportar `api/artefacto_pipeline.pkl` (versión 2.0.0).
- App: `sector_empleo` pasa a `visible_fields` como selectbox; ajustar `app_core.py` y tests.
- Docs: nueva DEC-012 (no reescribir DEC-010), README, Diseño_Transformaciones, app/README.

## Constraints
- Artefacto y schema se despliegan juntos (si no, /predict da 500).
- Reentrenar con las versiones pinneadas del `.venv` (pandas 3.0.5, sklearn 1.9.0, cloudpickle 3.1.2).
- Las métricas deben quedar iguales (ROC-AUC validación ~0.709) porque el mapeo no cambia.
- Preservar BOM en los `payload.json`.
- Hay cambios sin commitear previos del usuario en README.md, decisions.md, Diseño_Transformaciones.md y Notebooks/04: no mezclarlos en mis commits.
- Regla del proyecto: no hacer build tras los cambios. El reentrenamiento es parte del entregable de B.

## Execution config
- TDD: no configurado (fuente: ninguna). Se corren checks funcionales.
- Runner app: `.venv\Scripts\python.exe -m unittest discover -s 07_despliegue\app\tests -v`
- Delivery strategy: ask-on-risk. Forecast < 400 líneas de código sin el pkl; sin push ni PR.

## Tasks
- [x] T1 Backend: reentrenamiento + schema + payloads + validación externa + reexportar artefacto 2.0.0. Ruta: delegada (writer). Commit: 5715b63
- [x] T2 App: selectbox `sector_empleo`, `app_core.py`, `design_spec.json`, tests. Ruta: delegada (writer). Commit: 845d0c7
- [ ] T3 Docs: DEC-012, README, Diseño_Transformaciones, app/README. Ruta: delegada (writer). Commit: _pendiente_

## Acceptance criteria
- `POST /predict` con `sector_empleo` válido responde 200; con sector inválido o sin sector, 422.
- Métricas de validación externa sin cambios.
- Tests de la app en verde.
- El dashboard envía el sector elegido.

## Trigger evidence
Mapeo de 4+ archivos delegado a un explorador (Mapping trigger). T1-T3 tocan 2+ archivos no triviales cada una (Writer trigger).

## Progress / verification evidence
- T1 (5715b63): reentreno PD ROC-AUC 0.7077, EAD MAE 0.1606, LGD MAE 0.0903 (idénticos a 1.3.0). Validación externa PD 0.7087, `git diff 06_resultados` vacío. TestClient: /predict 200; sector inválido, ausente, null, `empleo` viejo -> 422. Pickle carga sin 04_scripts. Spot check parent: commit con 7 archivos, sin atribución, cambios previos del usuario sin tocar.
- T2 (845d0c7): 19 tests OK (spot check parent re-ejecutado: Ran 19, OK). Payload por defecto lleva `sector_empleo` y no `empleo`. Sin `streamlit run` contra API en vivo.
- Review T1/T2: RDD off (global) -> sin revisión nativa; `review assess` figura unassessable por untracked, irrelevante con RDD off.
- Pendiente menor: `FastAPI(version="1.0.0")` en api/main.py no se subió a 2.0.0.

## Next step
Delegar T2 (app).
