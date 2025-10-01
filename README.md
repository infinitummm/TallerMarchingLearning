# Regresión Lineal - Implementación ML de regesion lineal y comparación con scikit-learn


## 1) Implementación manual 
- Archivo: `linear_models.py`
- Clase: `SimpleLinearRegression`
- Calcula `b1` y `b0` usando las fórmulas estadísticas convencionales (sumatorias).
- Interfaz: `fit(x, y)`, `predict(x)`, `score(x, y)`, `params()`, `summary()`.
- Está escrita para aprender y comprender paso a paso cómo se obtienen los parámetros.

## 2) Wrapper de scikit-learn 
- Archivo: `linear_models.py`
- Clase: `SKLearnLinearRegressionWrapper`
- Envuelve `sklearn.linear_model.LinearRegression` con la misma interfaz POO.
- Permite comparar fácilmente coeficiente, intercepto y R² con la implementación manual.

## Ejemplos incluidos
- En la sección `if __name__ == "__main__":` del archivo se muestra:
  - Entrenamiento con el dataset pequeño (10 puntos).
  - Comparación de parámetros y R² entre la implementación manual y sklearn 
  - Intento de cargar `diamonds.csv` si está presente para entrenar/medir con un dataset real.

## Archivos adjuntos

- ComparacionCodigo.pdf:
  - Contiene el analisis y la comparacion de los dos codigos entregados.
- ML-Regresion.py:
  - Contiene el codigo del modelo de ML para una regresion lineal.
