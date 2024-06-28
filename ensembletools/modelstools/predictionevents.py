import logging

import datetime
import numpy as np
import pandas as pd


from dbbinance.fetcher import check_convert_to_datetime, convert_timeframe_to_freq, get_timedelta_kwargs
from ensembletools.modelstools import RawPredictionHistory
from ensembletools.modelstools import PredictionTracker
from ensembletools.modelstools import get_agg_dict

__version__ = 0.058

logger = logging.getLogger()


class PredictionsEvents(PredictionTracker):
    def __init__(self,
                 symbol: str,
                 market: str,
                 raw_ph_obj: RawPredictionHistory = None,
                 ):
        super().__init__(symbol, market, raw_ph_obj)
        pass

    @staticmethod
    def get_arg_preds(_df) -> np.ndarray:
        arg_preds = np.argmax(_df[[0, 1]].values, axis=-1)
        return arg_preds

    def _distillate_events(self, events_collect: list) -> dict or None:
        events_list: list = []
        for model_uuid, _df in events_collect:
            result_df = _df.set_index('predict_time')
            arg_events = self.get_arg_preds(result_df)
            if not np.any(arg_events):
                continue
            _arg_events_idx = list(np.where(arg_events == 1.)[0])
            if _arg_events_idx:
                events_df = result_df.iloc[_arg_events_idx]
            else:
                continue
            events_list.append((model_uuid, events_df))
        return events_list

    def get_sorted_distilled_events(self,
                                    start_datetime: datetime.datetime or str,
                                    end_datetime: datetime.datetime or str):
        sorted_data = self.get_sorted_events(start_datetime, end_datetime)
        sorted_distilled_data: dict = {}
        for type_dir, events_collect in sorted_data.items():
            distilled_events_collect = self._distillate_events(events_collect)
            sorted_distilled_data.update({type_dir: distilled_events_collect})
        return sorted_distilled_data

    def get_events_for_timeframe(self,
                                 start_datetime: datetime.datetime or str,
                                 end_datetime: datetime.datetime or str,
                                 timeframe: str):
        frequency = convert_timeframe_to_freq(timeframe)

        timedelta_kwargs = get_timedelta_kwargs(timeframe, timeframe)
        start_datetime = check_convert_to_datetime(start_datetime, utc_aware=False)

        end_datetime = check_convert_to_datetime(end_datetime, utc_aware=False)
        if start_datetime.tzinfo != end_datetime.tzinfo:
            start_datetime = start_datetime.replace(tzinfo=None)
            end_datetime = end_datetime.replace(tzinfo=None)

        predict_target_range = pd.DataFrame(pd.Series(pd.date_range(start_datetime,
                                                                    end_datetime,
                                                                    freq=frequency)),
                                            columns=['predict_time'])

        trgt_start_datetime = start_datetime + datetime.timedelta(**timedelta_kwargs)
        trgt_end_datetime = end_datetime + datetime.timedelta(**timedelta_kwargs)
        predict_target_range['target_time'] = pd.Series(pd.date_range(trgt_start_datetime,
                                                                      trgt_end_datetime,
                                                                      freq=frequency))

        events_data = self.get_sorted_events(start_datetime, end_datetime)
        new_events_data: dict = {}
        new_cards_data: dict = {}
        for type_dir, events_collect in events_data.items():
            new_events_collect: list = []
            for model_uuid, _df in events_collect:
                _df['power'] = np.ones((_df.shape[0],))
                agg_dict = get_agg_dict(list(_df.columns))
                new_df = pd.concat([predict_target_range, _df], axis=0)
                new_df = new_df.set_index('predict_time', drop=False)
                new_df = new_df.sort_index()
                new_df = new_df.resample(frequency, label='right', closed='right', origin='start').agg(agg_dict)
                new_events_collect.append((model_uuid, new_df))
                new_cards_data.update({model_uuid: self.get_model_card(model_uuid)})
            new_events_data.update({type_dir: new_events_collect})
        return new_events_data, new_cards_data
