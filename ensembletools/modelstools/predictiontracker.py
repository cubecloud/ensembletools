import copy
import logging

import datetime
import numpy as np
import pandas as pd

from collections import OrderedDict
from typing import List, Dict, Union, Optional


from dbbinance.fetcher import CacheManager
from dbbinance.fetcher import check_convert_to_datetime, convert_timeframe_to_freq, get_timedelta_kwargs
from dbbinance.fetcher import Constants
from ensembletools.modelstools import RawPredictionHistory
from ensembletools.modelstools import get_raw_ph_obj

__version__ = 0.060

logger = logging.getLogger()


def rolling_first(rows):
    return rows.values[0]


def rolling_last(rows):
    return rows.values[-1]


records_agg_dict = {'predict_time': 'last', 'target_time': 'last', 'num': np.nanmean, 'power': 'sum'}

records_rolling_agg_dict = {'num': np.nanmean, 'power': 'sum'}


def get_agg_dict(use_cols):
    actual_agg_dict = OrderedDict()
    for col_name in use_cols:
        if col_name in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            actual_agg_dict.update({col_name: records_agg_dict.get('num', None)})
        else:
            actual_agg_dict.update({col_name: records_agg_dict.get(col_name, None)})
    return actual_agg_dict


def get_rolling_agg_dict(use_cols):
    actual_agg_dict = OrderedDict()
    for col_name in use_cols:
        if col_name in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            actual_agg_dict.update({col_name: records_rolling_agg_dict.get('num', None)})
        else:
            actual_agg_dict.update({col_name: records_rolling_agg_dict.get(col_name, None)})
    return actual_agg_dict


cache_manager_obj = CacheManager()


class PredictionTracker:
    CM = cache_manager_obj

    def __init__(self, symbol: str, market: str, raw_ph_obj: RawPredictionHistory = None):
        """
        Initializes an instance of the class.

        Parameters:
            symbol (str): The symbol for the market.
            market (str): The market for the symbol.
            raw_ph_obj (RawPredictionHistory, optional): An instance of the RawPredictionHistory class. Defaults to None.

        Returns:
            None
        """

        if raw_ph_obj is None:
            raw_ph_obj: RawPredictionHistory = get_raw_ph_obj()

        self.raw_ph_obj: RawPredictionHistory = raw_ph_obj
        self.raw_pred_table_name = self.raw_ph_obj.prepare_table_name(symbol, market, )
        self.market = market
        self.symbol = symbol
        self.models_uuid_to_filter_list = []
        self.active_models_uuid_list = []

    def set_models_uuid_to_filter(self, models_uuid_filter_list: list):
        self.models_uuid_to_filter_list = models_uuid_filter_list

    def reset_models_uuid_filter(self):
        self.models_uuid_to_filter_list = []

    def get_all_models_uuid_list(self):
        return self.raw_ph_obj.get_unique_ids_list(self.raw_pred_table_name, self.symbol)

    def set_active_models_uuid(self, active_models_uuid_list):
        self.active_models_uuid_list = active_models_uuid_list

    def filter_models_uuid(self, models_uuid_to_filter_list: list = ()):
        all_models_uuid_list = self.get_all_models_uuid_list()
        if models_uuid_to_filter_list:
            self.set_active_models_uuid(list(set(all_models_uuid_list).difference(set(models_uuid_to_filter_list))))
        else:
            self.set_active_models_uuid(
                list(set(all_models_uuid_list).difference(set(self.models_uuid_to_filter_list))))

    def get_all_models_raw_predictions(self,
                                       start_datetime: datetime.datetime or str,
                                       end_datetime: datetime.datetime or str
                                       ) -> Dict or None:
        start_datetime = check_convert_to_datetime(start_datetime)
        end_datetime = check_convert_to_datetime(end_datetime)

        models_pred: dict = {}

        for model_uuid in self.active_models_uuid_list:
            pred_data = self.raw_ph_obj.get_predictions_by_predict_time(predicts_table_name=self.raw_pred_table_name,
                                                                        symbol=self.symbol,
                                                                        model_uuid=model_uuid,
                                                                        predict_start_datetime=start_datetime,
                                                                        predict_end_datetime=end_datetime)
            if pred_data is not None:
                models_pred.update({model_uuid: pred_data})
        return models_pred

    @staticmethod
    def prepare_dict_df(records_dict):
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

    @staticmethod
    def indexed_dict_df(records_dict, start_datetime, end_datetime):
        models_dict_df: dict = {}
        index = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
        for k, v in records_dict.items():
            records = pd.DataFrame(v)
            records = records.drop(columns='model_uuid')
            records = records.sort_values(by='predict_time', ascending=True).set_index('predict_time')
            pred_data = pd.DataFrame(records['predict_data'].to_list(), index=records.index).astype(float)
            records = pd.concat([records['target_time'], pred_data], axis=1)
            records = records.reindex(index, method=None)
            models_dict_df[k] = records.add_prefix(k + '#')
        return models_dict_df

    @staticmethod
    def custom_reindex(records_df, index, fillna: Union[int, float] = 0):
        columns_to_fill = ['power', 0, 1]
        records_df = records_df.reindex(index, method=None)
        records_df[columns_to_fill] = records_df[columns_to_fill].fillna(fillna)
        return records_df

    def load_predicted_data(self,
                            start_datetime: Union[datetime.datetime, str],
                            end_datetime: Union[datetime.datetime, str],
                            utc_aware=False) -> Dict[str, pd.DataFrame]:

        start_datetime = check_convert_to_datetime(start_datetime, utc_aware=utc_aware)
        end_datetime = check_convert_to_datetime(end_datetime, utc_aware=utc_aware)

        channels_data: dict = {}

        for model_uuid in self.active_models_uuid_list:
            predicted_data = self.raw_ph_obj.get_predictions_by_predict_time(
                predicts_table_name=self.raw_pred_table_name,
                symbol=self.symbol,
                model_uuid=model_uuid,
                predict_start_datetime=start_datetime,
                predict_end_datetime=end_datetime)
            if predicted_data:
                channels_data.update(self.indexed_dict_df({model_uuid: predicted_data},
                                                          start_datetime,
                                                          end_datetime))
        return channels_data

    @staticmethod
    def one_min_records_to_df(records_lst,
                              start_datetime,
                              end_datetime,
                              ) -> pd.DataFrame:
        index = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
        records_df = pd.DataFrame(records_lst).drop(columns='model_uuid')
        records_df['power'] = np.ones((records_df.shape[0],))
        target_time_diff = (pd.to_datetime(records_df['target_time'].iloc[0]) - (
            pd.to_datetime(records_df['predict_time'].iloc[0])))
        records_df = records_df.sort_values(by='predict_time', ascending=True).set_index('predict_time')
        pred_data_df = pd.DataFrame(records_df['predict_data'].to_list(), index=records_df.index).astype(float)
        records_df = pd.concat([records_df['power'], pred_data_df], axis=1)
        records_df = PredictionTracker.custom_reindex(records_df, index)
        records_df['power'] = records_df['power'].astype(float)
        records_df['target_time'] = records_df.index + target_time_diff
        return records_df

    @staticmethod
    def powered_df(records_lst,
                   start_datetime,
                   end_datetime,
                   timeframe: Union[str, None] = None,
                   discretization: Union[str, None] = None) -> pd.DataFrame:
        index = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
        records_df = pd.DataFrame(records_lst).drop(columns='model_uuid')
        records_df['power'] = np.ones((records_df.shape[0],))
        target_time_diff = (pd.to_datetime(records_df['target_time'].iloc[0]) - (
            pd.to_datetime(records_df['predict_time'].iloc[0])))
        records_df = records_df.sort_values(by='predict_time', ascending=True).set_index('predict_time')
        pred_data_df = pd.DataFrame(records_df['predict_data'].to_list(), index=records_df.index).astype(float)
        records_df = pd.concat([records_df['power'], pred_data_df], axis=1)

        if timeframe is not None and timeframe != '1m':
            records_df = PredictionTracker.custom_reindex(records_df, index, fillna=np.nan)
            assert discretization is not None, f'Error: discretization is not set - None'
            if timeframe == discretization:
                agg_dict = get_agg_dict(list(records_df.columns))
                frequency = convert_timeframe_to_freq(timeframe)
                records_df = records_df.resample(frequency, label='right', closed='right', origin='end').agg(
                    agg_dict)
            else:
                cols = list(records_df.columns)
                agg_dict = get_rolling_agg_dict(cols)
                frequency = Constants.binsizes.get(discretization)
                records_df = records_df.rolling(frequency, min_periods=1).agg(agg_dict)
                records_df = records_df.iloc[::Constants.binsizes.get(timeframe)]
        else:
            records_df = PredictionTracker.custom_reindex(records_df, index)
        records_df['power'] = records_df['power'].astype(float)
        records_df['target_time'] = records_df.index + target_time_diff
        return records_df

    def get_predicted_data(self, model_uuid, start_datetime, end_datetime):
        predicted_data = self.raw_ph_obj.get_predictions_by_predict_time(
            predicts_table_name=self.raw_pred_table_name,
            symbol=self.symbol,
            model_uuid=model_uuid,
            predict_start_datetime=start_datetime,
            predict_end_datetime=end_datetime)
        return predicted_data

    def load_model_predicted_data(self,
                                  model_uuid: str,
                                  start_datetime: Union[datetime.datetime, str],
                                  end_datetime: Union[datetime.datetime, str],
                                  timeframe: Optional[str] = None,
                                  discretization: Optional[str] = None,
                                  utc_aware: bool = False,
                                  cached: bool = False,
                                  ) -> Optional[pd.DataFrame]:

        def get_powered_df():
            pwrd_df = None
            predicted_data = None
            if cached:
                cache_key = self.CM.get_cache_key(symbol=self.symbol, market=self.market, model_uuid=model_uuid,
                                                  start_datetime=extended_start_datetime, end_datetime=end_datetime,
                                                  timeframe=timeframe, discretization=discretization, data_type='RAW')
                if cache_key in self.CM.cache.keys():
                    predicted_data = self.CM.cache[cache_key]
                    logger.debug(f"{self.__class__.__name__}: Return RAW predicted cached data {model_uuid}")
                else:
                    for key in self.CM.cache.keys():
                        if len(key) == 8:
                            if (key[6][1] == self.symbol) and (extended_start_datetime >= key[5][1]) and (
                                    end_datetime <= key[2][1]) and (model_uuid == key[4][1]) and (
                                    timeframe == key[7][1]) and ('RAW' == key[0][1]) and (
                                    self.market == key[3][1]) and (discretization == key[1][1]):
                                predicted_data = self.CM.cache[key]
                                msg = f"{self.__class__.__name__}: Return RAW predicted cached data {model_uuid}"
                                logger.debug(msg)
                                break
                    if predicted_data is None:
                        predicted_data = self.get_predicted_data(model_uuid, extended_start_datetime, end_datetime)
                        self.CM.update_cache(key=cache_key, value=copy.deepcopy(predicted_data))
            else:
                predicted_data = self.get_predicted_data(model_uuid, extended_start_datetime, end_datetime)

            if predicted_data:
                pwrd_df = self.powered_df(records_lst=predicted_data,
                                          start_datetime=extended_start_datetime,
                                          end_datetime=end_datetime,
                                          timeframe=timeframe,
                                          discretization=discretization)
            return pwrd_df

        start_datetime = check_convert_to_datetime(start_datetime, utc_aware=utc_aware)
        # decrement start_datetime by longer length (timeframe or discretization), to get correct data
        if discretization is None:
            discretization = timeframe
        if Constants.binsizes.get(timeframe, None) > Constants.binsizes.get(discretization, None):
            time_delta = timeframe
        else:
            time_delta = discretization
        timedelta_kwargs = get_timedelta_kwargs(time_delta)
        extended_start_datetime = start_datetime - datetime.timedelta(**timedelta_kwargs)

        end_datetime = check_convert_to_datetime(end_datetime, utc_aware=utc_aware)

        if cached:
            cache_key = self.CM.get_cache_key(symbol=self.symbol, market=self.market, model_uuid=model_uuid,
                                              start_datetime=extended_start_datetime, end_datetime=end_datetime,
                                              timeframe=timeframe)
            if cache_key in self.CM.cache.keys():
                powered_df = self.CM.cache[cache_key]
                logger.debug(f"{self.__class__.__name__}: Return cached data. {model_uuid}")
            else:
                powered_df = get_powered_df()
                self.CM.update_cache(cache_key, powered_df)
        else:
            powered_df = get_powered_df()

        return powered_df.loc[start_datetime:].copy(deep=True) if powered_df is not None else None

    @staticmethod
    def some_dict_update(k: any, list_value: any, some_dict: dict):
        v = some_dict.get(k, None)
        if v is None:
            v = []
        v.append(list_value)
        some_dict.update({k: v})
        return some_dict

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





