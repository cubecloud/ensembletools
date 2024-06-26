import datetime
import logging
from dateutil.relativedelta import relativedelta
from dbbinance.fetcher.datautils import get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import ceil_time, floor_time
from ensembletools.modelstools.predictionstore import PredictionsTracker
from ensembletools.modelstools.predictionstore import get_raw_ph_obj


logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('test_predictiontracker.log')
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
_start_datetime = floor_time(_start_datetime, _timeframe)

pt = PredictionsTracker(symbol='BTCUSDT', market='spot', raw_ph_obj=raw_db_obj)

models_uuid = pt.get_all_models_uuid_list()
pt.set_active_models_uuid(models_uuid)

logger.debug(f"Models UUID's list:\n{models_uuid}")

logger.debug(f"Load model data for model_uuid: {models_uuid[0]}, with 'start_datetime-end_datetime': {_start_datetime}-{_end_datetime} and 'timeframe': {_timeframe}")

_df = pt.load_model_predicted_data(model_uuid=models_uuid[0],
                                   start_datetime=_start_datetime,
                                   end_datetime=_end_datetime,
                                   utc_aware=False,
                                   timeframe=_timeframe)

if _df is not None:
    logger.debug(f"df:\n{_df.to_string()}")
else:
    logger.debug(f"df: None")

