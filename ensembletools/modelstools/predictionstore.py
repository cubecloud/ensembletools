import copy
import logging

import datetime
import numpy as np
import pandas as pd

from psycopg2 import sql
from psycopg2.extras import execute_values
from collections import OrderedDict
from typing import List, Dict, Union, Optional

from dbbinance.config.configpostgresql import ConfigPostgreSQL
from dbbinance.fetcher.cachemanager import CacheManager
from ensembletools.modelstools.modelcard_v2 import ModelCard
# from ensembletools.modelstools.calculations import *
from dbbinance.fetcher.datautils import check_convert_to_datetime, convert_timeframe_to_freq, get_timedelta_kwargs
from dbbinance.fetcher.datafetcher import Constants
from ensembletools.modelstools.predictionrecord import PredictionRecord
from dbbinance.fetcher.sqlbase import SQLMeta, handle_errors

__version__ = 0.049

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


class ModelsSQLDatabase(SQLMeta):
    models_table_name = "models_cards_base"

    def __init__(self, host, database, user, password, max_conn=10):
        super().__init__(host, database, user, password, max_conn)

        self.create_models_table()

    # def is_table_exists(self, table_name: str):
    #     with self.models_conn.cursor() as cur:
    #         cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)", (table_name,))
    #         result = cur.fetchone()
    #     return result[0]

    @handle_errors
    def create_models_table(self):
        """
        Create the cards table in the database if it doesn't exist.
        """
        conn = self.conn  # Used for optimizing calls of @property
        if not self.is_table_exists(table_name=self.models_table_name):
            with conn.cursor() as cur:
                ct = sql.SQL('''
                            CREATE TABLE IF NOT EXISTS {} (
                                model_uuid TEXT PRIMARY KEY,
                                model_name TEXT,
                                model_type TEXT,
                                model_activator_key TEXT,
                                model_activator_value TEXT,
                                symbol TEXT,
                                market TEXT,
                                interval TEXT,
                                power_trend FLOAT,
                                time_window_hours INTEGER,
                                timeframes_period INTEGER,
                                target_steps INTEGER,
                                use_pooltype TEXT,
                                start_period TEXT,
                                end_period TEXT,
                                fixed_end_period TEXT
                            )
                        ''').format(sql.Identifier(self.models_table_name))
                cur.execute(ct)
            conn.commit()
            logger.debug(f"{self.__class__.__name__} #{self.idnum}: created table '{self.models_table_name}'")
        else:
            logger.debug(f"{self.__class__.__name__} #{self.idnum}: table '{self.models_table_name}' exists")
        conn.close()

    @handle_errors
    def save_model_card(self, card: ModelCard):
        """
        Save a Card object to the cards table

        Args:
            card: the Card object to be saved

        Returns:
            None:
        """
        conn = self.conn  # Used for optimizing calls of @property
        with conn.cursor() as cur:
            # Insert card data into table
            insert = sql.SQL('''
                INSERT INTO {models_table} (
                                model_uuid,
                                model_name,
                                model_type,
                                model_activator_key,
                                model_activator_value,
                                symbol,
                                market,
                                interval,
                                power_trend,
                                time_window_hours,
                                timeframes_period,
                                target_steps,
                                use_pooltype,
                                start_period,
                                end_period,
                                fixed_end_period
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_uuid) DO UPDATE SET
                     model_name = excluded.model_name,
                     model_type = excluded.model_type,
                     model_activator_key = excluded.model_activator_key,
                     model_activator_value = excluded.model_activator_value,
                     symbol = excluded.symbol,
                     market = excluded.market,
                     interval = excluded.interval,
                     power_trend = excluded.power_trend,
                     time_window_hours = excluded.time_window_hours,
                     timeframes_period = excluded.timeframes_period,
                     target_steps = excluded.target_steps,
                     use_pooltype = excluded.use_pooltype,                     
                     start_period = excluded.start_period,
                     end_period = excluded.end_period,
                     fixed_end_period = excluded.fixed_end_period
            ''').format(models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))

            cur.execute(insert, (card.model_uuid,
                                 card.model_name,
                                 card.model_type,
                                 card.model_activator_key,
                                 card.model_activator_value,
                                 card.symbol,
                                 card.market,
                                 card.interval,
                                 card.power_trend,
                                 card.time_window_hours,
                                 card.timeframes_period,
                                 card.target_steps,
                                 card.use_pooltype,
                                 card.start_period,
                                 card.end_period,
                                 card.fixed_end_period))
        conn.commit()

    def get_card(self, model_uuid: str) -> ModelCard or None:
        """
        Retrieve a Card object from the cards table by model_uuid

        Args:
            model_uuid (str):  the model_uuid in the ModelCard object

        Returns:
             card (object):         the Card object for the specified model_uuid
        """
        conn = self.conn  # Used for optimizing calls of @property
        with conn.cursor() as cur:
            # Query for card data by model_uuid
            select = sql.SQL('SELECT * FROM {models_table} WHERE model_uuid = %s').format(
                models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))

            cur.execute(select, (model_uuid,))

            row = cur.fetchone()
        conn.close()

        if row is None:
            return None
        # Create and return Card object from row data
        card_data = dict(zip([desc[0] for desc in cur.description], row))

        card = ModelCard(**card_data)

        return card

    @handle_errors
    def get_models_unique_ids(self) -> List[str]:
        """
        Retrieve a list of model_uuid values from the model_data table

        Returns:
            List[str]:  models_ids
        """
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT model_uuid FROM {self.models_table_name}")
            rows = cur.fetchall()
        models_unique_ids = [row[0] for row in rows]
        return models_unique_ids

    @handle_errors
    def is_model_unique_id_exists(self, model_uuid) -> bool:
        with self.conn.cursor() as cur:
            query = f"SELECT COUNT(*) FROM {self.models_table_name} WHERE model_uuid = '{model_uuid}'"
            cur.execute(query)
            result = cur.fetchone()
        return result[0] > 0


class PredictionSQLDatabase(SQLMeta):
    def __init__(self, host, database, user, password, max_conn=10):
        super().__init__(host, database, user, password, max_conn)
        self.initialized_tables: list = []

    def check_table(self, table_name):
        if not (table_name in self.initialized_tables):
            if not self.is_table_exists(table_name):
                self.create_table(table_name)
            self.initialized_tables.append(table_name)

    @handle_errors
    def create_table(self, table_name: str = "spot_data"):
        """
        Create the predictions table in the database if it doesn't exist.
        """
        conn = self.conn
        with self.conn.cursor() as cur:
            create_table = sql.SQL('''
                CREATE TABLE IF NOT EXISTS {predict_table} (
                    id SERIAL PRIMARY KEY,
                    model_uuid TEXT REFERENCES {models_table} (model_uuid),
                    predict_time TIMESTAMP,
                    target_time TIMESTAMP,
                    predict_data NUMERIC[] NOT NULL)
            ''').format(predict_table=sql.Identifier(table_name),
                        models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))
            cur.execute(create_table)
        conn.commit()

    @handle_errors
    def record_exists(self, table_name: str, prediction_record: PredictionRecord):
        # logger.debug(f"{self.__class__.__name__}: check if record exist in the table '{table_name}'")
        with self.conn.cursor() as cur:
            select = sql.SQL('''
                SELECT COUNT(*) FROM {table_name}
                WHERE model_uuid = %s AND predict_time = %s ''').format(table_name=sql.Identifier(table_name))

            cur.execute(select, (prediction_record.model_uuid, prediction_record.predict_time))
            result = cur.fetchone()
        return result[0] > 0

    def records_exist(self, table_name: str, prediction_records: list):
        # Prepare a list of model_uuid and predict_time for the SQL query
        model_uuids = [record.model_uuid for record in prediction_records]
        predict_times = [record.predict_time for record in prediction_records]

        with self.conn.cursor() as cur:
            select = sql.SQL('''
                SELECT model_uuid, predict_time, COUNT(*) 
                FROM {table_name} 
                WHERE model_uuid = ANY(%s) AND predict_time = ANY(%s)
                GROUP BY model_uuid, predict_time
            ''').format(table_name=sql.Identifier(table_name))

            cur.execute(select, (model_uuids, predict_times))
            result = cur.fetchall()

        # Prepare a dictionary where key is a tuple (model_uuid, predict_time) and value is the count
        result_dict = {(row[0], row[1]): row[2] for row in result}

        # Prepare the final result list
        results = [int(result_dict.get((record.model_uuid, record.predict_time), 0) > 0) for record in
                   prediction_records]

        return results

    @handle_errors
    def get_predict_data(self, table_name: str, prediction_record: PredictionRecord):
        # logger.debug(f"{self.__class__.__name__}: check if record exist in the table '{table_name}'")
        with self.conn.cursor() as cur:
            select = sql.SQL('''
                SELECT * FROM {table_name}
                WHERE model_uuid = %s AND predict_time = %s AND target_time = %s  
            ''').format(table_name=sql.Identifier(table_name))

            cur.execute(select, (prediction_record.model_uuid,
                                 prediction_record.predict_time,
                                 prediction_record.target_time,
                                 ))
            record = cur.fetchone()
        result = PredictionRecord(*record[1:])
        result.predict_data = np.array(result.predict_data).astype(float)
        return result

    @handle_errors
    def insert_record(self, table_name: str, prediction_record: PredictionRecord):
        conn = self.conn
        with conn.cursor() as cur:
            insert = sql.SQL('''
                INSERT INTO {table_name} (model_uuid, predict_time, target_time, predict_data)
                VALUES (%s, %s, %s, %s)
            ''').format(table_name=sql.Identifier(table_name))

            cur.execute(insert, (prediction_record.model_uuid,
                                 prediction_record.predict_time,
                                 prediction_record.target_time,
                                 prediction_record.predict_data))
        conn.commit()
        # logger.debug(f"{self.__class__.__name__}: record inserted to table '{table_name}'")

    @handle_errors
    def insert_many_records(self, table_name: str, records: List[tuple,]):
        conn = self.conn
        with conn.cursor() as cur:
            insert_values = sql.SQL(
                '''INSERT INTO {} (model_uuid, predict_time, target_time, predict_data) VALUES %s''').format(
                sql.Identifier(table_name))
            execute_values(cur, insert_values, records)
        conn.commit()

    def save_record(self, table_name: str, prediction_record: PredictionRecord,
                    duplicate_compare: bool = False, verbose: bool = False):
        self.check_table(table_name)
        if not self.record_exists(table_name, prediction_record):
            self.insert_record(table_name, prediction_record)
            if verbose:
                logger.debug(f"{self.__class__.__name__} #{self.idnum}: record saved to table '{table_name}'. "
                             f"Datetime: {prediction_record.predict_time}.")
            result = 1
        else:
            if verbose:
                logger.debug(f"'{self.__class__.__name__}' #{self.idnum}: record exist in table '{table_name}'. "
                             f"Datetime: {prediction_record.predict_time} - skip insertion")
            if duplicate_compare:
                saved_record = self.get_predict_data(table_name, prediction_record)
                logger.warning(f"'{self.__class__.__name__}' #{self.idnum}: Table name: '{table_name}'. "
                               f"Compare records predict_data. "
                               f"To save: {prediction_record.predict_time} {prediction_record.predict_data}")
                logger.warning(f"'{self.__class__.__name__}' #{self.idnum}: Table name: '{table_name}'. "
                               f"Compare records predict_data. "
                               f"Saved: {saved_record.predict_time} {saved_record.predict_data}")
            result = 0
        return result

    @handle_errors
    def get_all_records(self, table_name) -> List[PredictionRecord]:
        with self.conn.cursor() as cur:
            select_query = f"SELECT * FROM {table_name}"
            cur.execute(select_query)
            rows = cur.fetchall()
        prediction_records = [PredictionRecord(*row[1:]) for row in rows]
        return prediction_records

    @handle_errors
    def get_records(self, table_name, start_datetime, end_datetime):
        with self.conn.cursor() as cur:
            select_statement = f"""
                SELECT * FROM {table_name} WHERE predict_time >= '{start_datetime}' AND predict_time <= '{end_datetime}'
            """
            cur.execute(select_statement)
            rows = cur.fetchall()
        predictions = [PredictionRecord(*row[1:]) for row in rows]
        return predictions


class RawPredictionHistory(ModelsSQLDatabase, PredictionSQLDatabase):

    def __init__(self, host, database, user, password, table_name_suffix='_data'):
        super().__init__(host, database, user, password)
        SQLMeta.count -= 1  # decrease number cos we have two parents with counted idnum
        PredictionSQLDatabase.__init__(self, host, database, user, password)
        self.idnum = int(SQLMeta.count)
        self.table_name_suffix = table_name_suffix

    def prepare_table_name(self, symbol: str = 'BTCUSDT', market: str = None, ) -> str:
        if market is None:
            market = "spot"
        base_table_name = f"{market}{self.table_name_suffix}"
        table_name = f"{base_table_name}_{symbol}_predictions".lower()
        return table_name

    def save_prediction(self,
                        prediction_record: PredictionRecord,
                        model_card: ModelCard,
                        table_name: Union[str, None] = None,
                        ) -> int:
        """
        Args:
            table_name:
            prediction_record (PredictionRecord):
            model_card (Card):

        Returns:
            None:
        """
        if table_name is None:
            table_name = self.prepare_table_name(model_card.symbol, model_card.market)
        self.check_table(table_name)

        if not self.is_model_unique_id_exists(model_card.model_uuid):
            self.save_model_card(model_card)

        result = self.save_record(table_name=table_name, prediction_record=prediction_record)
        return result

    def _collect_not_duped_records(self,
                                   records: List[PredictionRecord,],
                                   table_name: Union[str, None] = None) -> List[PredictionRecord,]:
        not_duped: list = []
        result = self.records_exist(table_name, records)

        for prediction_record, is_dupe in zip(records, result):
            if is_dupe == 0:
                not_duped.append(prediction_record)
        return not_duped

    def save_all_predictions(self, records: List[PredictionRecord,], model_card: ModelCard,
                             table_name: str or None = None,
                             ):
        """
        Args:
            records (List[PredictionRecord,]:
            model_card (Card):
            table_name:

        Returns:
            None:
        """
        total_len = len(records)
        if table_name is None:
            table_name = self.prepare_table_name(model_card.symbol, model_card.market)
        self.check_table(table_name)

        if not self.is_model_unique_id_exists(model_card.model_uuid):
            self.save_model_card(model_card)
        else:
            logger.debug(f"{self.__class__.__name__} #{self.idnum}: Start checking for dupes records '{table_name}'.")
            records = self._collect_not_duped_records(records, table_name)
            logger.debug(f"{self.__class__.__name__} #{self.idnum}: Stop checking for dupes records '{table_name}'.")

        data_records: list = []
        for row in records:
            data_records.append((row.model_uuid, row.predict_time, row.target_time, row.predict_data))
        self.insert_many_records(table_name, data_records)
        logger.info(
            f"{self.__class__.__name__} #{self.idnum}: {total_len}/{len(records)} records saved to '{table_name}'.")

    def get_unique_ids_list(self,
                            predicts_table_name: str,
                            symbol: str,
                            ) -> List[str]:
        """
        Retrieve PredictionRecord objects from the database by model_uuid, symbol

        Args:
            predicts_table_name (str):  name of the table to retrieve predictions from
            symbol (str):          symbol of the PredictionRecord objects to retrieve

        Returns:
            List[str]:                  list of unique_ids_list
        """

        with self.conn.cursor() as cur:
            select_query = sql.SQL('''
             SELECT DISTINCT {predicts_table}.model_uuid
             FROM {predicts_table}
             INNER JOIN {models_table}
             ON {predicts_table}.model_uuid = {models_table}.model_uuid
             WHERE {models_table}.symbol = %s
             ''').format(predicts_table=sql.Identifier(predicts_table_name),
                         models_table=sql.Identifier(ModelsSQLDatabase.models_table_name)
                         )
            cur.execute(select_query, (symbol,))
            results = cur.fetchall()
        if results:
            return [result[0] for result in results]
        else:
            return []

    def get_predictions_with_unique_id(self,
                                       predicts_table_name: str,
                                       model_uuid: str
                                       ) -> List[PredictionRecord] or None:
        """
        Retrieve PredictionRecord objects from the database with model_uuid.

        Args:
            predicts_table_name (str):          name of the table to retrieve predictions from
            model_uuid (str):                   model_uuid of the PredictionRecord objects to retrieve

        Returns:
            List[PredictionRecord] or None:     list of prediction records
        """

        with self.conn.cursor() as cur:
            # Query for prediction data by model_uuid, symbol, and datetime range
            select = sql.SQL('''
                SELECT p.* FROM {predicts_table} p
                JOIN {models_table} c ON p.model_uuid = c.model_uuid
                WHERE p.model_uuid = %s
            ''').format(predicts_table=sql.Identifier(predicts_table_name),
                        models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))
            cur.execute(select, (model_uuid,))
            rows = cur.fetchall()

        predictions = []
        if rows:
            for row in rows:
                # Create and append PredictionRecord object from row data
                prediction_data = dict(zip([desc[0] for desc in cur.description[1:]], row[1:]))
                prediction = PredictionRecord(**prediction_data)
                prediction.predict_data = np.array(prediction.predict_data).astype(float)
                predictions.append(prediction)
        else:
            predictions = None
        return predictions

    def get_predictions_by_predict_time(self,
                                        predicts_table_name: str,
                                        model_uuid: str,
                                        symbol: str,
                                        predict_start_datetime: datetime.datetime,
                                        predict_end_datetime: datetime.datetime,
                                        ) -> List[PredictionRecord]:
        """
        Retrieve PredictionRecord objects from the database by model_uuid, symbol, and datetime range.

        Args:
            predicts_table_name (str):                  name of the table to retrieve predictions from
            model_uuid (str):                           model_uuid of the PredictionRecord objects to retrieve
            symbol (str):                               symbol of the PredictionRecord objects to retrieve
            predict_start_datetime (datetime.datetime): start of the datetime range to retrieve predictions for
            predict_end_datetime (datetime.datetime):   end of the datetime range to retrieve predictions for

        Returns:
            List[PredictionRecord]:     list of prediction records
        """

        with self.conn.cursor() as cur:
            # Query for prediction data by model_uuid, symbol, and datetime range
            select = sql.SQL('''
                SELECT p.* FROM {predicts_table} p
                JOIN {models_table} c ON p.model_uuid = c.model_uuid
                WHERE p.model_uuid = %s AND c.symbol = %s AND p.predict_time BETWEEN %s AND %s
            ''').format(predicts_table=sql.Identifier(predicts_table_name),
                        models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))

            cur.execute(select, (model_uuid, symbol, predict_start_datetime, predict_end_datetime))

            rows = cur.fetchall()

        predictions = []
        if rows:
            for row in rows:
                # Create and append PredictionRecord object from row data
                prediction_data = dict(zip([desc[0] for desc in cur.description[1:]], row[1:]))
                prediction = PredictionRecord(**prediction_data)
                prediction.predict_data = np.array(prediction.predict_data).astype(float)
                predictions.append(prediction)
        else:
            predictions = None
        return predictions

    def get_predictions_by_target_time(self,
                                       predicts_table_name: str,
                                       model_uuid: str,
                                       symbol: str,
                                       target_start_datetime: datetime.datetime,
                                       target_end_datetime: datetime.datetime,
                                       ) -> List[PredictionRecord]:
        """
        Retrieve PredictionRecord objects from the database by model_uuid, symbol, and datetime range.

        Args:
            predicts_table_name (str):                  name of the table to retrieve predictions from
            model_uuid (str):                           model_uuid of the PredictionRecord objects to retrieve
            symbol (str):                               symbol of the PredictionRecord objects to retrieve
            target_start_datetime (datetime.datetime):  start of the datetime range to retrieve predictions for
            target_end_datetime (datetime.datetime):    end of the datetime range to retrieve predictions for

        Returns:
            List[PredictionRecord]:     list of prediction records
        """
        with self.conn.cursor() as cur:
            # Query for prediction data by model_uuid, symbol, and datetime range
            select = sql.SQL('''
                SELECT p.* FROM {predicts_table} p
                JOIN {models_table} c ON p.model_uuid = c.model_uuid
                WHERE p.model_uuid = %s AND c.symbol = %s AND p.target_time BETWEEN %s AND %s
            ''').format(predicts_table=sql.Identifier(predicts_table_name),
                        models_table=sql.Identifier(ModelsSQLDatabase.models_table_name))

            cur.execute(select, (model_uuid, symbol, target_start_datetime, target_end_datetime))

            rows = cur.fetchall()

        predictions = []
        if rows:
            for row in rows:
                # Create and append PredictionRecord object from row data
                prediction_data = dict(zip([desc[0] for desc in cur.description[1:]], row[1:]))
                prediction = PredictionRecord(**prediction_data)
                prediction.predict_data = np.array(prediction.predict_data).astype(float)
                predictions.append(prediction)
        else:
            predictions = None
        return predictions


models_type_dict: dict = {"buy": "binary_markers",
                          "sell": "binary_markers",
                          }

cache_manager_obj = CacheManager()


class PredictionsTracker:
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
            records_df = PredictionsTracker.custom_reindex(records_df, index, fillna=np.nan)
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
            records_df = PredictionsTracker.custom_reindex(records_df, index)
        records_df['power'] = records_df['power'].astype(float)
        records_df['target_time'] = records_df.index + target_time_diff
        return records_df

    def load_model_predicted_data(self,
                                  model_uuid: str,
                                  start_datetime: Union[datetime.datetime, str],
                                  end_datetime: Union[datetime.datetime, str],
                                  timeframe: Optional[str] = None,
                                  discretization: Optional[str] = None,
                                  utc_aware: bool = False,
                                  cached: bool = False,
                                  ) -> Optional[pd.DataFrame]:

        def get_predicted_data():
            predicted_data = self.raw_ph_obj.get_predictions_by_predict_time(
                predicts_table_name=self.raw_pred_table_name,
                symbol=self.symbol,
                model_uuid=model_uuid,
                predict_start_datetime=extended_start_datetime,
                predict_end_datetime=end_datetime)
            return predicted_data

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
                        if (len(key) == 8) and (key[6][1] == self.symbol) and (
                                extended_start_datetime >= key[5][1]) and (
                                end_datetime <= key[2][1]) and (model_uuid == key[4][1]) and (
                                timeframe == key[7][1]) and ('RAW' == key[0][1]) and (self.market == key[3][1]) and (
                                discretization == key[1][1]):
                            predicted_data = self.CM.cache[key]
                            msg = f"{self.__class__.__name__}: Return RAW predicted cached data {model_uuid}"
                            logger.debug(msg)
                            break
                    if predicted_data is None:
                        predicted_data = get_predicted_data()
                        self.CM.update_cache(key=cache_key, value=copy.deepcopy(predicted_data))
            else:
                predicted_data = get_predicted_data()

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


class PredictionsEvents(PredictionsTracker):
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


def get_raw_ph_obj(table_name_suffix='_data') -> RawPredictionHistory:
    db_kwargs_ = dict(host=ConfigPostgreSQL.HOST,
                      database=ConfigPostgreSQL.DATABASE,
                      user=ConfigPostgreSQL.USER,
                      password=ConfigPostgreSQL.PASSWORD,
                      )

    raw_db_ = RawPredictionHistory(**db_kwargs_, table_name_suffix=table_name_suffix)
    return raw_db_
