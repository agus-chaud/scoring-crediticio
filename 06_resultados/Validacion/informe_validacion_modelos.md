# Validación externa del artefacto de scoring

La evaluación se ejecutó sobre `02_datos/02_Validacion/validacion.pkl`, separado antes del reentrenamiento.

## Resultados

- Registros evaluados: 59833
- Casos de incumplimiento: 7006
- PD ROC-AUC: 0.7025
- EAD MAE, solo incumplimientos: 0.1536
- LGD MAE, solo incumplimientos: 0.0885

## Interpretación

Estas métricas muestran comportamiento sobre datos no usados para entrenar. No sustituyen la validación de negocio, el análisis de sesgos, la calibración ni el monitoreo posterior.
