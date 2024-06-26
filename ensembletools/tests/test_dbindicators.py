import datetime
import logging
from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from ensembletools.indicators import DbIndicator, IndicatorLoaded
from ensembletools.modelstools import PredictionTracker
from ensembletools.modelstools import get_raw_ph_obj

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
_discretization = '15m'

_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), _timeframe)
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = ceil_time(_start_datetime, _timeframe)

pt = PredictionTracker(symbol='BTCUSDT', market='spot', raw_ph_obj=raw_db_obj)

models_uuid = pt.get_all_models_uuid_list()
pt.set_active_models_uuid(models_uuid)

logger.debug(f"{__name__}: Models UUID's list:\n{models_uuid}")

logger.debug(
    f"{__name__}: Load model data for model_uuid: {models_uuid[0]}, with 'start_datetime-end_datetime': {_start_datetime}-{_end_datetime}")

ind = IndicatorLoaded(models_uuid[0], prediction_tracker_obj=pt)
ind._timeframe = _timeframe
ind.discretization = _discretization

logger.debug(
    f"{__name__}: Preload indicator data with start_datetime-end_datetime': {_start_datetime} - {_end_datetime}, timeframe: {_timeframe}, discretization: {_discretization}")
ind.preload_indicator(_start_datetime, _end_datetime)

logger.debug(f"{__name__}: Indicator uuid: {models_uuid[0]}")
logger.debug(f"{__name__}: Indicator name: {ind.name}")

use_columns = [0, 1, 'power']
logger.debug(f"{__name__}: Set show columns: {use_columns}")
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"{__name__}: Indicator data: \n{_df}")

use_columns = [1]
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"{__name__}: Set only __main__ column to show: {use_columns}")
logger.debug(f"{__name__}: Indicator data: \n{_df}")

logger.debug(f"{__name__}: Setting different timeframe and discretization: ")
_show_period = '3h'
_timeframe = '5m'
_discretization = '15m'

_timedelta_kwargs = get_timedelta_kwargs(_show_period, current_timeframe=_timeframe)
_end_datetime = ceil_time(datetime.datetime.utcnow(), '1m')
_start_datetime = _end_datetime - relativedelta(**_timedelta_kwargs)
_start_datetime = ceil_time(_start_datetime, '1m')

ind.timeframe = _timeframe
ind.discretization = _discretization

logger.debug(
    f"{__name__}: Preload indicator data with start_datetime-end_datetime': {_start_datetime} - {_end_datetime}, timeframe: {_timeframe}, discretization: {_discretization}")

ind.preload_indicator(_start_datetime, _end_datetime)

use_columns = [0, 1, 'power', 'target_time']
ind.columns = use_columns
_df = ind.prediction_show
logger.debug(f"{__name__}: Set only __main__ columns to show: {use_columns}")
logger.debug(f"{__name__}: Indicator data: \n{_df}")

use_columns = [0, 1, 'power', 'prediction_time']
ind.columns = use_columns
logger.debug(f"{__name__}: Set 'index_type' = target_time to use 'target_time column as index")
ind.index_type = 'target_time'
ind.preload_indicator(_start_datetime, _end_datetime)
_df = ind.prediction_show
logger.debug(f"{__name__}: Set only __main__ columns to show: {use_columns}")
logger.debug(f"{__name__}: Indicator data: \n{_df}")
