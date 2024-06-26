import datetime
import numpy as np
from dataclasses import dataclass

__version__ = 0.022


@dataclass
class PredictionRecord:
    model_uuid: str = None  # model_uuid
    predict_time: datetime.datetime = None  # datetime of prediction
    target_time: datetime.datetime = None  # datetime of predict_target
    predict_data: np.ndarray = None  # predict data
