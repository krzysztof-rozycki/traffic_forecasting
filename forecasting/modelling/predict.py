import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def predict(model, X: np.ndarray):
    return model.predict(X)


def evaluate(y_pred: np.ndarray, y_true: pd.Series):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred)
    }
