import logging

import datetime
import numpy as np
import pandas as pd

from typing import List, Dict

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

    def prepare_dict_df(self, records_dict):
        """
        Prepare a dictionary of dataframes.
        Args:
            records_dict (dict): A dictionary of records, where the keys are the model_UUID names and the values are lists of records.

        Returns:
            dict: A dictionary where the keys are the record names and the values are pandas DataFrames containing the records.
        """
        models_dict_df: dict = {}
        for k, v in records_dict.items():
            records = pd.DataFrame(v)
            records = records.drop(columns='model_uuid')
            pred_data = pd.DataFrame(records['predict_data'].to_list(), index=records.index).astype(float)
            records = pd.concat([records, pred_data], axis=1)
            records = records.drop(columns=['predict_data'])
            records = records.sort_values(by='predict_time', ascending=True)
            models_dict_df[k] = records
        return models_dict_df

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

    def get_sorted_events(self,
                          start_datetime: datetime.datetime or str,
                          end_datetime: datetime.datetime or str,
                          ) -> dict:
        """
        Getting all events in datetime range 'start_datetime'-'end_datetime'
        with models_uuid's

        Args:
            start_datetime:
            end_datetime:
        """
        events_data: dict = {}
        self.set_active_models_uuid(self.get_all_models_uuid_list())
        records_dict = self.get_all_models_raw_predictions(start_datetime=start_datetime,
                                                           end_datetime=end_datetime)
        if records_dict is None:
            return events_data

        records_dict = self.prepare_dict_df(records_dict)
        sorted_cards = self.get_sorted_cards(list(records_dict.keys()))
        for model_uuid, events_df in records_dict.items():
            for type_dir, cards_ in sorted_cards.items():
                models_uuids = [card.model_uuid for card in cards_]
                if model_uuid in models_uuids:
                    events_data = self.some_dict_update(type_dir, (model_uuid, events_df), events_data)
        return events_data

    def get_sorted_cards(self, models_uuids) -> Dict[str, List]:
        """
        Get sorted models cards dictionary
        Args:
            models_uuids (list):        list of models uuids for sort

        Returns:
            sorted_cards_dict (dict):   sorted cards
        """
        cards_type_dict: dict = {}
        for model_uuid in models_uuids:
            model_card = self.get_model_card(model_uuid=model_uuid)
            if model_card.model_activator_value == 'plus' and model_card.model_type == 'classification':
                cards_type_dict = self.some_dict_update('plus', model_card, cards_type_dict)
            elif model_card.model_activator_value == 'minus' and model_card.model_type == 'classification':
                cards_type_dict = self.some_dict_update('minus', model_card, cards_type_dict)
            elif model_card.model_type == 'regression':
                cards_type_dict = self.some_dict_update('regression', model_card, cards_type_dict)
            else:
                cards_type_dict = self.some_dict_update('unknown', model_card, cards_type_dict)
        return cards_type_dict

    @staticmethod
    def some_dict_update(k: any, list_value: any, some_dict: dict):
        v = some_dict.get(k, None)
        if v is None:
            v = []
        v.append(list_value)
        some_dict.update({k: v})
        return some_dict

    def get_model_card(self, model_uuid: str):
        model_card = self.raw_ph_obj.get_card(model_uuid=model_uuid)
        return model_card

    def get_models_predictions(self,
                               start_datetime: datetime.datetime or str,
                               end_datetime: datetime.datetime or str,
                               models_uuid_list: list = (),
                               ):
        self.set_active_models_uuid(models_uuid_list)
        models_pred = self.get_all_models_raw_predictions(start_datetime, end_datetime)
        self.reset_models_uuid_filter()
        return models_pred
