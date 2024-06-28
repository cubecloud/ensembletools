import logging

import datetime
import numpy as np

from psycopg2 import sql
from psycopg2.extras import execute_values
from typing import List, Union

from ensembletools.modelstools import ModelCard
from ensembletools.modelstools.predictionrecord import PredictionRecord
from dbbinance.fetcher import SQLMeta, handle_errors
from dbbinance.config import ConfigPostgreSQL

__version__ = 0.053

logger = logging.getLogger()


class ModelsSQLDatabase(SQLMeta):
    models_table_name = "models_cards_base"

    def __init__(self, host, database, user, password, max_conn=10):
        super().__init__(host, database, user, password, max_conn)

        self.create_models_table()

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


def get_raw_ph_obj(table_name_suffix='_data') -> RawPredictionHistory:
    db_kwargs_ = dict(host=ConfigPostgreSQL.HOST,
                      database=ConfigPostgreSQL.DATABASE,
                      user=ConfigPostgreSQL.USER,
                      password=ConfigPostgreSQL.PASSWORD,
                      )

    raw_db_ = RawPredictionHistory(**db_kwargs_, table_name_suffix=table_name_suffix)
    return raw_db_
