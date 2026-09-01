## ESTADO ACTUAL DEL PROYECTO

**Fases completadas**: reconstrucción de pipelines, artefacto de scoring, API local y cliente Streamlit demostrativo.

**Artefacto**: `07_despliegue/artefacto_pipeline.pkl`

**Motor de producción**: `07_despliegue/02_produccion_scoring.py` y `07_despliegue/api/scoring.py`

**API canónica**: `07_despliegue/api`; `POST /predict` recibe una lista JSON `[{...}]` y devuelve una lista con `score_pd`, `score_ead`, `score_lgd` y `perdida_esperada_relativa`.

**Validación local de API**: desde `07_despliegue/api`, `uvicorn api.main:app --app-dir ..`.

**Deploy de API en Render**: root `07_despliegue/api`; build `pip install -r requirements.txt`; start `uvicorn api.main:app --host 0.0.0.0 --port $PORT --app-dir ..`; `PYTHON_VERSION=3.13.7`.

**Interfaz Streamlit**: `07_despliegue/app/app.py` consume sólo la API canónica. No carga modelos ni artefactos, conserva los valores crudos y muestra resultados y sensibilidad exclusivamente demostrativos.

**Siguiente paso**: ejecutar la verificación manual de accesibilidad/responsividad y, si corresponde, desplegar primero la API y luego la interfaz Streamlit configurando `API_BASE_URL`.