# Deploy manual en Render

## Antes de abrir Render

1. Crear un repositorio GitHub para este proyecto y subir el contenido completo,
   incluyendo `07_despliegue/api/artefacto_pipeline.pkl`.
2. No subir datos sensibles ni credenciales.

## Configuración del servicio

En Render, crear un **Web Service** conectado al repositorio y usar:

| Campo | Valor |
|---|---|
| Root Directory | `07_despliegue/api` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT --app-dir ..` |
| Environment Variable | `PYTHON_VERSION=3.13.7` |

## Verificación posterior

Cuando Render indique que el deploy terminó, abrir:

`https://<tu-servicio>.onrender.com/health`

Debe responder:

```json
{"status":"ok"}
```

Después abrir `https://<tu-servicio>.onrender.com/docs` y probar `POST /predict`
con el contenido de `test_payload.json`.

## Si falla

Copiar el log de inicio completo de Render. Los errores más probables son una
dependencia faltante, una versión de Python distinta o que no se haya subido el
archivo `artefacto_pipeline.pkl`.
