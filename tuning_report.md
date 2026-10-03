# Informe de ajuste de hiperparámetros — StreamLoop Churn Prediction

## 1. Elección de métrica

El problema de negocio de StreamLoop es **detectar clientes que van a cancelar** para poder ofrecerles una retención antes de que se vayan. Perder a un cliente sin intentar retenerlo (falso negativo) es mucho más caro que ofrecerle un descuento a alguien que no lo necesitaba (falso positivo).

Por eso se eligió **Recall (sensibilidad)** como métrica principal de optimización y evaluación. El accuracy por sí solo es engañoso en este dataset desbalanceado (73.5% No Churn, 26.5% Yes Churn).

---

## 2. Metodología

| Fase | Método | Alcance |
|---|---|---|
| **Línea base** | RandomForest con hiperparámetros por defecto | Punto de comparación inicial |
| **Búsqueda amplia** | `RandomizedSearchCV` — 60 iteraciones sobre 4.500 combinaciones posibles | Identificar zonas prometedoras |
| **Búsqueda precisa** | `GridSearchCV` — 96 combinaciones acotadas alrededor de la zona ganadora | Refinar los mejores valores |

Ambas búsquedas usaron:
- Validación cruzada de **5 folds** sobre el **train set** únicamente
- `scoring='recall'` (alineado con la prioridad de negocio)
- `refit=True` para reentrenar automáticamente el mejor modelo en todo el train
- El **test set** se mantuvo intacto hasta la evaluación final

---

## 3. Hiperparámetros finales

```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,    # usa todas las features
    'bootstrap': False
}
```

---

## 4. Resultados comparativos

### Validación cruzada (train, 5 folds)

| Modelo | Recall medio CV | Desviación estándar |
|---|---|---|
| Línea base (por defecto) | 0.4803 | ±0.0300 |
| RandomizedSearch (mejor) | 0.5164 | ±0.0117 |
| **GridSearch (elegido)** | **0.5184** | **±0.0127** |

### Evaluación en test set (única evaluación)

| Métrica | Línea base | Modelo final | Cambio |
|---|---|---|---|
| **Recall** | **0.4893** | **0.5294** | **+4.01 pp** |
| Accuracy | 0.7921 | 0.7580 | −3.41 pp |
| Precision | 0.6421 | 0.5455 | −9.67 pp |
| F1-score | 0.5554 | 0.5373 | −1.81 pp |
| ROC AUC | 0.6954 | 0.6850 | −1.04 pp |

La mejora en Recall se logró a costa de perder algo de precisión (más falsos positivos), lo cual es aceptable según la prioridad de negocio: es preferible ofrecer retención a algunos clientes que no la necesitan que dejar que un cliente que sí iba a cancelar se vaya sin ser detectado.

---

## 5. Trade-off considerado: promedio vs estabilidad

Al analizar `cv_results_` del GridSearch, aparecieron dos candidatos principales:

| Candidato | Recall medio | Desviación std | Coef. variación |
|---|---|---|---|
| **A** (max_depth=8, leaf=2) — máximo promedio | 0.5231 | 0.0572 | 10.9% |
| **B ← ELEGIDO** (max_depth=10, leaf=1) — balance | 0.5184 | 0.0127 | 2.4% |
| **C** (max_depth=10, leaf=1, split=5) — máxima estabilidad | 0.5110 | 0.0074 | 1.5% |

**El candidato A** tiene el mejor promedio (0.5231), pero su desviación entre folds es **4.5× mayor** que la del candidato B (0.0572 vs 0.0127). En producción esto implicaría un rendimiento impredecible: el modelo podría detectar 58% de churners en un mes y 45% al siguiente dependiendo de la composición del batch.

**Se eligió el candidato B** porque:
- Sacrifica solo **0.005 puntos** de Recall promedio
- Su coeficiente de variación es 4.5× menor (2.4% vs 10.9%)
- Es más fiable para planificar campañas de retención consistentes

---

## 6. Matriz de confusión — Modelo final en test

```
              Pred: No Churn    Pred: Churn
Real: No Churn      870             165
Real: Sí Churn      176             198
```

- **VP = 198**: clientes que cancelaron y fueron detectados → se les puede ofrecer retención
- **FN = 176**: clientes que cancelaron y NO fueron detectados → pérdida evitable (el mayor coste)
- **FP = 165**: clientes que no cancelaban pero recibirían oferta → coste asumible
- **VN = 870**: clientes correctamente clasificados como no churners

---

