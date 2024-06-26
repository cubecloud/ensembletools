import datetime
import numpy as np
from dataclasses import dataclass, asdict

__version__ = 0.021


@dataclass
class PredictionRecord:
    model_uuid: str = None  # model_uuid
    predict_time: datetime.datetime = None  # datetime of prediction
    target_time: datetime.datetime = None  # datetime of predict_target
    predict_data: np.ndarray = None  # predict data
