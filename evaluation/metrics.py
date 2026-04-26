import numpy as np


def mean_absolute_error(true_values, predicted_values):
    """
    Computes Mean Absolute Error (MAE)

    MAE measures the average absolute difference between
    the true values and the predicted values.

    Lower MAE = better predictions
    """
    # absolute difference between prediction and ground truth
    absolute_errors = np.abs(true_values - predicted_values)

    # average of those errors
    mae_value = np.mean(absolute_errors)

    return mae_value


def mean_absolute_percentage_error(true_values, predicted_values):
    """
    Computes Mean Absolute Percentage Error (MAPE)

    MAPE measures the average percentage error between
    predicted and true values.

    Lower MAPE = better predictions
    """

    # avoid division by zero by replacing 0s with a very small value
    safe_true_values = np.where(true_values == 0, 1e-8, true_values)

    # compute percentage errors
    percentage_errors = np.abs((safe_true_values - predicted_values) / safe_true_values)

    # convert to percentage scale
    mape_value = np.mean(percentage_errors) * 100

    return mape_value


def r2_score(true_values, predicted_values):
    """
    Computes R-squared

    R^2 measures how well predictions explain the variance
    in the true data.

    R^2 = 1 -> perfect prediction
    R^2 = 0 -> no better than mean
    R^2 < 0 -> worse than predicting the mean
    """

    # residual sum of squares (difference between predictions and actual values)
    residual_sum_of_squares = np.sum((true_values - predicted_values) ** 2)

    # total sum of squares (variance in the true data)
    total_sum_of_squares = np.sum((true_values - np.mean(true_values)) ** 2)

    # edge case: if there is no variance in true values
    if total_sum_of_squares == 0:
        return 0

    # compute R^2 score
    r2_value = 1 - (residual_sum_of_squares / total_sum_of_squares)

    return r2_value