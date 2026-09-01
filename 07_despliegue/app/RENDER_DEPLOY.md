# Deploy manual en Render — app Streamlit

La API canónica ya está en `https://scoring-crediticio.onrender.com`. Este
servicio es la interfaz y la consume por HTTP.

## Antes de abrir Render

1. El código ya está en `origin/main` (commit `1d89dcb` o posterior).
2. `07_despliegue/app/.streamlit/config.toml` debe estar versionado (tema azul).
3. No subir datos sensibles ni credenciales.

## Configuración del servicio

En Render, crear un **Web Service** conectado a `agus-chaud/AA_scoring-crediticio` y usar:

| Campo | Valor |
|---|---|
| Root Directory | `07_despliegue/app` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Environment Variable | `PYTHON_VERSION=3.13.7` |
| Environment Variable | `API_BASE_URL=https://scoring-crediticio.onrender.com` |

Sin `API_BASE_URL` la app apunta a `http://127.0.0.1:8000` y ninguna consulta
funciona en Render.

## Verificación posterior

1. Esperar a que Render marque el deploy como `Live` y anotar el commit que
   muestra en **Events** / **Deploys**: debe ser `1d89dcb` (o el `HEAD` actual
   de `main`), no uno anterior.
2. Abrir `https://<servicio-app>.onrender.com`. Debe cargar el tablero: panel de
   filtros a la izquierda y estado vacío a la derecha.
3. Tocar **Calcular riesgo** con los valores por defecto. La primera llamada
   puede tardar ~1 min por el cold start de la API. Resultado esperado con los
   defaults: pérdida esperada relativa ≈ 11,5 %, etiqueta **Riesgo alto**.
4. Confirmar que NO aparecen los avisos técnicos viejos y que la píldora dice
   "Riesgo bajo / medio / alto" (no "umbral demostrativo").

## Redeploy cuando haya cambios nuevos

Con **Auto-Deploy** activado, cada push a `main` dispara un build. Si está
desactivado: Render → el servicio → **Manual Deploy → Deploy latest commit**.
Siempre verificar en **Events** que el hash desplegado coincide con
`git rev-parse --short HEAD` local.

## Si falla

Copiar el log de build/arranque completo. Lo más probable: una dependencia que
no resuelve, `PYTHON_VERSION` distinta, o `API_BASE_URL` sin setear.
