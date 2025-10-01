from typing import List, Tuple
import math


import matplotlib.pyplot as plt


try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class SimpleLinearRegression:
  

    def __init__(self):
        self.b0 = 0.0
        self.b1 = 0.0
        self.fitted = False

    def _mean(self, values: List[float]) -> float:
        return sum(values) / len(values)

    def fit(self, x: List[float], y: List[float]) -> None:
       
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        n = len(x)
        mean_x = self._mean(x)
        mean_y = self._mean(y)

      
        sum_xx = 0.0  
        sum_xy = 0.0  
        for xi, yi in zip(x, y):
            dx = xi - mean_x
            dy = yi - mean_y
            sum_xx += dx * dx
            sum_xy += dx * dy

        if math.isclose(sum_xx, 0.0):
            raise ValueError("Variance of x is zero; can't fit line")

        self.b1 = sum_xy / sum_xx
        self.b0 = mean_y - self.b1 * mean_x
        self.fitted = True

    def predict(self, x: List[float]) -> List[float]:
        """Devuelve lista de predicciones para los valores x."""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        return [self.b0 + self.b1 * xi for xi in x]

    def score(self, x: List[float], y: List[float]) -> float:
        """
        R^2: coeficiente de determinación.
        R^2 = 1 - SS_res / SS_tot
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        y_pred = self.predict(x)
        mean_y = self._mean(y)
        ss_res = 0.0
        ss_tot = 0.0
        for yi, ypi in zip(y, y_pred):
            ss_res += (yi - ypi) ** 2
            ss_tot += (yi - mean_y) ** 2
        if math.isclose(ss_tot, 0.0):
            return 0.0
        return 1 - (ss_res / ss_tot)

    def params(self) -> Tuple[float, float]:
        return self.b0, self.b1

    def summary(self):
        if not self.fitted:
            print("Model not fitted.")
            return
        print(f"Intercept (b0): {self.b0}")
        print(f"Slope (b1): {self.b1}")


class SKLearnLinearRegressionWrapper:
   

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn no encontrado. Esta clase no funcionará sin sklearn.")
        self.model = LinearRegression() if SKLEARN_AVAILABLE else None
        self.fitted = False

    def fit(self, x: List[float], y: List[float]):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn no está instalado")
     
        X = [[val] for val in x]
        Y = [val for val in y]
        self.model.fit(X, Y)
        self.fitted = True

    def predict(self, x: List[float]) -> List[float]:
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        X = [[val] for val in x]
        return [float(v) for v in self.model.predict(X)]

    def score(self, x: List[float], y: List[float]) -> float:
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        X = [[val] for val in x]
        return float(self.model.score(X, y))

    def params(self) -> Tuple[float, float]:
        """Devuelve (intercept, slope)"""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        return float(self.model.intercept_), float(self.model.coef_[0])

    def summary(self):
        if not self.fitted:
            print("Model not fitted.")
            return
        intercept, slope = self.params()
        print(f"Intercept (b0): {intercept}")
        print(f"Slope (b1): {slope}")



if __name__ == "__main__":
   
    carats = [0.3, 0.41, 0.75, 0.91, 1.2, 1.31, 1.5, 1.74, 1.96, 2.21]
    carats_scaled = [c * 1000 for c in carats]  # igual que hiciste
    price = [339, 561, 2760, 2763, 2809, 3697, 4022, 4677, 6147, 6535]

 
    simple = SimpleLinearRegression()
    simple.fit(carats_scaled, price)
    simple.summary()
    preds_simple = simple.predict(carats_scaled)
    print("R2 (manual):", simple.score(carats_scaled, price))

    if SKLEARN_AVAILABLE:
        skl = SKLearnLinearRegressionWrapper()
        skl.fit(carats_scaled, price)
        skl.summary()
        print("R2 (sklearn):", skl.score(carats_scaled, price))
    else:
        print("sklearn no disponible: salteando la comparación con sklearn.")

    xs_plot = sorted(carats_scaled)
    ys_simple = simple.predict(xs_plot)
    try:
        plt.figure(figsize=(6, 5))
        plt.scatter(carats_scaled, price, label="Datos (train)", color="black")
        plt.plot(xs_plot, ys_simple, label="Ajuste SimpleLinearRegression", color="red", linewidth=2)
        if SKLEARN_AVAILABLE:
            ys_skl = skl.predict(xs_plot)
            plt.plot(xs_plot, ys_skl, label="Ajuste sklearn", color="green", linewidth=1.5)
        plt.xlabel("Carat (scaled)")
        plt.ylabel("Price")
        plt.title("Comparación modelos lineales")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("No se pudo graficar:", e)


    try:
        import pandas as pd
        df = pd.read_csv("diamonds.csv")
      
        X_all = df["price"].tolist()
        y_all = df["carat"].tolist()
        if SKLEARN_AVAILABLE:
            skl2 = SKLearnLinearRegressionWrapper()
            skl2.fit(X_all, y_all)
            print("\nResultados sobre diamonds.csv (sklearn):")
            skl2.summary()
            print("R2 (diamonds):", skl2.score(X_all, y_all))
        else:
            print("sklearn no disponible: no se ajustó diamonds.csv")
    except FileNotFoundError:
        print("diamonds.csv no encontrado en el directorio actual. Omitiendo prueba con archivo grande.")
    except Exception as e:
        print("Error intentando usar diamonds.csv:", e)

