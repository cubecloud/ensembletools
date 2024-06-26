import numpy as np

__version__ = 0.003


def scale_prediction(_predict,
                     _predict_range: tuple = (0, 1),
                     ) -> np.array:
    """
    Prepare scaled predictions for better understand _the range_
        predict (np.array): predictions
    Returns:
        scaled_predict (np.array):
    """
    scaled_predict = _predict.copy()
    for ix in range(_predict.shape[-1]):
        scaled_predict[:, ix] = np.interp(_predict[:, ix],
                                          (_predict[:, ix].min(), _predict[:, ix].max()),
                                          _predict_range
                                          )
    return scaled_predict


def calc_signal_with_treshold(cat_predict, threshold=0.5) -> np.array:
    interp_pred = scale_prediction(cat_predict)
    temp_arr = interp_pred > threshold
    trend_pred = np.zeros(temp_arr.shape[0])
    for ix in range(temp_arr.shape[0]):
        if temp_arr[ix].any():
            trend_pred[ix] = 1.0 if temp_arr[ix].argmax() else 0.0
        else:
            trend_pred[ix] = 0.0
    return trend_pred


def calc_class_predictions(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # Scale each channel of the input data separately using interpolation
    scaled_probs = np.zeros_like(probs)
    for i in range(probs.shape[1]):
        if probs[:, i].max() < 0.5:
            scaled_probs[:, i] = probs[:, i]
        else:
            scaled_probs[:, i] = np.interp(probs[:, i], (probs[:, i].min(), probs[:, i].max()), (0, 1))

    # Calculate the class predictions
    return (scaled_probs[:, 1] > threshold).astype(int)
