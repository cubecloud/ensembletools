import time
import logging
import datetime

import pandas as pd

from dbbinance.fetcher.datautils import Constants
from dataclasses import dataclass
from ensembletools.modelstools import ModelCard
from ensembletools.modelstools import PredictionTracker
from dbbinance.fetcher.datautils import check_convert_to_datetime, get_timedelta_kwargs
from dateutil.relativedelta import relativedelta

__version__ = 0.019

logger = logging.getLogger()


@dataclass
class ModelAlgoParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class DbIndicator:

    def __init__(self, model_uuid: str, prediction_tracker_obj: PredictionTracker = None):
        self.model_uuid: str = model_uuid
        self.indicator_id: int = 0
        self.pt_obj = prediction_tracker_obj
        self.model_card: ModelCard = self.pt_obj.raw_ph_obj.get_card(self.model_uuid)
        self.__timeframe = self.model_card.interval
        self.__discretization = self.model_card.interval
        self.__current_datetime: datetime.datetime or None = None
        self._indicator_data: pd.DataFrame or pd.Series or None = None
        self.__algo_params: ModelAlgoParams or None = None
        self.retry_counter: int = 0

    @property
    def algoparams(self):
        return self.__algo_params

    @algoparams.setter
    def algoparams(self, params: dict):
        self.__algo_params = ModelAlgoParams(**params)

    @property
    def power_trend(self):
        return self.model_card.power_trend

    @property
    def interval(self):
        return self.model_card.interval

    @property
    def target_steps(self):
        return self.model_card.target_steps

    @property
    def market(self):
        return self.model_card.market

    @property
    def symbol(self):
        return self.model_card.symbol

    @property
    def model_type(self):
        return self.model_card.model_type

    @property
    def direction(self):
        if self.model_card.model_activator_value == 'plus' and self.model_type == 'classification':
            return 'plus'
        elif self.model_card.model_activator_value == 'minus' and self.model_type == 'classification':
            return 'minus'
        else:
            return None

    @property
    def name(self):
        return f'{self.direction}_{self.power_trend}_{self.interval}_{self.discretization}'

    @property
    def discretization(self):
        return self.__discretization

    @discretization.setter
    def discretization(self, discretization: str):
        self.__discretization = discretization

    @property
    def timeframe(self):
        return self.__timeframe

    @timeframe.setter
    def timeframe(self, timeframe: str):
        self.__timeframe = timeframe

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, dt: datetime.datetime):
        self.__current_datetime = dt
        self._indicator_data = self._indicator(dt)

    @property
    def indicator(self) -> pd.Series:
        return self._indicator_data

    @property
    def prediction(self) -> float:
        return self._indicator_data[1]

    @property
    def power(self) -> float:
        return self._indicator_data['power']

    @property
    def target_time(self) -> datetime.datetime:
        return self._indicator_data['target_time']

    def _indicator(self, dt: datetime.datetime or str) -> pd.DataFrame or None:
        _timedelta_kwargs = get_timedelta_kwargs(self.discretization, current_timeframe='1m')
        _end_datetime = check_convert_to_datetime(dt, utc_aware=False)
        _start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
        _df = self.pt_obj.load_model_predicted_data(model_uuid=self.model_uuid,
                                                    start_datetime=_start_datetime,
                                                    end_datetime=_end_datetime,
                                                    timeframe=self.timeframe,
                                                    discretization=self.discretization,
                                                    )
        if _df is not None and _df.shape[0] > 0:
            _df['power'] = _df['power'] / Constants.binsizes[self.discretization]
            return _df.iloc[-1]
        else:
            if self.retry_counter == 2:
                self.retry_counter = 0
                return None
            else:
                self.retry_counter += 1
                #   retry 1 more time with recursion

                logger.warning(f'{self.__class__.__name__}: '
                               f'{self.model_uuid} received None, retry ({self.retry_counter})')
                time.sleep(self.retry_counter)
                return self._indicator(dt)


class IndicatorLoaded(DbIndicator):
    def __init__(self, model_uuid: str, prediction_tracker_obj: PredictionTracker = None):
        super().__init__(model_uuid, prediction_tracker_obj)
        self.__preloaded_data: pd.DataFrame or None = None
        self.__show_columns: list = [1]
        self.__index_type: str = 'prediction_time'

    @property
    def prediction_show(self) -> pd.DataFrame:
        if self.__index_type == 'target_time':
            _df = self.__preloaded_data.reset_index(drop=False).set_index('target_time', drop=False)
            cols = [col if col != 'index' else 'prediction_time' for col in _df.columns]
            _df.columns = cols
            _df.index.names = [None]
            return _df[self.__show_columns]
        else:
            return self.__preloaded_data[self.__show_columns]

    @property
    def power_show(self):
        return self.__preloaded_data['power']

    @property
    def columns(self):
        return self.__show_columns

    @columns.setter
    def columns(self, value: list):
        self.__show_columns = value

    @property
    def index_type(self):
        return self.__index_type

    @index_type.setter
    def index_type(self, value: str):
        assert value in ['prediction_time', 'target_time'], f'Error: unknown index type {self.__index_type}'
        self.__index_type = value

    @property
    def current_datetime(self):
        return self.__current_datetime

    @current_datetime.setter
    def current_datetime(self, dt: datetime.datetime):
        self.__current_datetime = dt
        try:
            self._indicator_data = self.__preloaded_data.loc[dt]
        except (AttributeError, KeyError):
            logger.warning(
                f'{self.__class__.__name__}: {self.model_uuid} Error: self.__preloaded_data(dt) {dt}, add `None`')
            self._indicator_data = None
            # _timedelta_kwargs = get_timedelta_kwargs(self.discretization, current_timeframe='1m')
            # self._indicator_data = pd.DataFrame(
            #     data=[[self.__current_datetime + relativedelta(**_timedelta_kwargs), 0, .0, .0, ]],
            #     columns=['target_time', 'power', 0, 1, ],
            #     index=[pd.to_datetime(self.__current_datetime)]).iloc[0]

    def preload_indicator(self,
                          _start_datetime: datetime.datetime or str,
                          _end_datetime: datetime.datetime or str,
                          deep_debug=False,
                          ):
        _start = check_convert_to_datetime(_start_datetime, utc_aware=False)
        _end = check_convert_to_datetime(_end_datetime, utc_aware=False)

        if self.__index_type == 'target_time':
            _timedelta_kwargs = get_timedelta_kwargs(self.model_card.target_steps,
                                                     current_timeframe=self.model_card.interval)
            _start = _start - relativedelta(**_timedelta_kwargs)
            _end = _end - relativedelta(**_timedelta_kwargs)

        _df = self.pt_obj.load_model_predicted_data(model_uuid=self.model_uuid,
                                                    start_datetime=_start,
                                                    end_datetime=_end,
                                                    timeframe=self.timeframe,
                                                    discretization=self.discretization,
                                                    cached=True)
        if deep_debug:
            logger.debug(f'{self.__class__.__name__}: Input: {self.model_uuid}  {_start_datetime} - {_end_datetime}')
            logger.debug(f'{self.__class__.__name__}: Preload: {self.model_uuid}  {_start} - {_end}')

        if _df is not None and _df.shape[0] > 0:
            self.__preloaded_data = _df.copy(deep=True)
            self.__preloaded_data['power'] = self.__preloaded_data['power'] / Constants.binsizes[self.discretization]
            if deep_debug:
                msg = (f'{self.__class__.__name__}: Loaded: {self.model_uuid}  '
                       f'{self.__preloaded_data.index[0]} - {self.__preloaded_data.index[-1]}')
                logger.debug(msg)
        else:
            self.__preloaded_data = None
        pass
