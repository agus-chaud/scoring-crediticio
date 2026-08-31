# Reconstrucción de preproducción

Se reconstruyó el flujo a partir de `Notebooks/01_ImportacionDatos.ipynb` a
`Notebooks/08_Preproduccion.ipynb` y de las tablas de entrenamiento existentes.

## Cadena conservada

1. Se separan los préstamos para entrenamiento y validación.
2. Se eliminan identificadores y texto libre; se completa la antigüedad laboral.
3. Se normalizan categorías poco frecuentes de vivienda y finalidad.
4. Se convierten las categorías y los valores numéricos al formato usado por los modelos.
5. Se entrenan PD para todos los préstamos, y EAD/LGD solo para casos de incumplimiento.

## Riesgos residuales

- Los notebooks históricos no fijaban semilla y ajustaban algunos transformadores antes de separar datos.
- El artefacto de producción corrige ambos puntos: usa `random_state=42` y encapsula la preparación dentro de cada pipeline.
- Los resultados EAD y LGD son proporciones condicionadas a incumplimiento; no son montos monetarios absolutos.
