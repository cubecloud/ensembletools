import datetime
import logging
from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from ensembletools.indicators.dbindicators import DbIndicator, IndicatorLoaded
from ensembletools.modelstools.predictionstore import PredictionsTracker
from ensembletools.modelstools.predictionstore import get_raw_ph_obj

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_dbindicators.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

raw_db_obj = get_raw_ph_obj()

_timeframe = '1h'
_show_period = '1d'
_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), _timeframe)
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = ceil_time(_start_datetime, _timeframe)

pt = PredictionsTracker(symbol='BTCUSDT', market='spot', raw_ph_obj=raw_db_obj)

models_uuid = pt.get_all_models_uuid_list()
pt.set_active_models_uuid(models_uuid)

logger.debug(f"Models UUID's list:\n{models_uuid}")

logger.debug(
    f"Load model data for model_uuid: {models_uuid[0]}, with 'start_datetime-end_datetime': {_start_datetime}-{_end_datetime} and 'timeframe': {_timeframe}")

ind = IndicatorLoaded(models_uuid[0], prediction_tracker_obj=pt)
logger.debug(f"Preload indicator data with start_datetime-end_datetime': {_start_datetime}-{_end_datetime}")
ind.preload_indicator(_start_datetime, _end_datetime)

logger.debug(f"Indicator uuid: {models_uuid[0]}")
logger.debug(f"Indicator name: {ind.name}")

use_columns = [0, 1, 'power']
logger.debug(f"Set show columns: {use_columns}")
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"Indicator data: \n{_df}")

use_columns = [1]
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"Set only __main__ column to show: {use_columns}")
logger.debug(f"Indicator data: \n{_df}")

logger.debug(f"Setting different timeframe and discretization: ")
_show_period = '3h'
ind.timeframe = '10m'
ind.discretization = '10m'
_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m')
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = floor_time(_start_datetime, '1m')
logger.debug(f"Preload indicator data with start_datetime-end_datetime': {_start_datetime}-{_end_datetime}")
ind.preload_indicator(_start_datetime, _end_datetime)

use_columns = [0, 1, 'power', 'target_time']
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"Set only __main__ columns to show: {use_columns}")
logger.debug(f"Indicator data: \n{_df}")

use_columns = [0, 1, 'power', 'prediction_time']
ind.columns = use_columns
logger.debug(f"Set 'index_type' = target_time to use 'target_time column as index")
ind.index_type = 'target_time'
_df = ind.prediction_show
logger.debug(f"Set only __main__ columns to show: {use_columns}")
logger.debug(f"Indicator data: \n{_df}")

