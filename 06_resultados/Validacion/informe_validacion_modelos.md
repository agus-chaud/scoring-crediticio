# Validación externa del artefacto de scoring

La evaluación se ejecutó sobre `02_datos/02_Validacion/validacion.pkl`, separado antes del reentrenamiento. Se aplican los mismos filtros de filas que en entrenamiento: solo préstamos con desenlace, ingresos <= 400.000 y dti distinto de 999.

## Resultados

- Registros evaluados: 35592
- Casos de incumplimiento: 7006
- PD ROC-AUC: 0.7087
- EAD MAE, solo incumplimientos: 0.1533
- LGD MAE, solo incumplimientos: 0.0885

## Interpretación

Estas métricas muestran comportamiento sobre datos no usados para entrenar. No sustituyen la validación de negocio, el análisis de sesgos, la calibración ni el monitoreo posterior.
