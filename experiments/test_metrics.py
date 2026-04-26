import numpy as np
from evaluation.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)

y_true = np.array([10, 20, 30, 40])
y_pred = np.array([12, 18, 29, 41])

print("MAE:", mean_absolute_error(y_true, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_true, y_pred))
print("R2:", r2_score(y_true, y_pred))