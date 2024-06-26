import yaml
import copy
import regex
import logging
import numpy as np
from sympy import *
from dataclasses import dataclass
from dbbinance.config.configpostgresql import ConfigPostgreSQL
from dbbinance.fetcher.constants import Constants
from dbbinance.fetcher.datafetcher import DataFetcher

__version__ = '0.0.14'

logger = logging.getLogger()


@dataclass
class Organizer:
    def __init__(self):
        self.name: str or None = None
        self.params: dict or None = None
        self.ftz_obj: object = None
        self.test_gen_obj: object = None
        # self.train_gen_obj: object = None
        # self.val_gen_obj: object = None
        self.model_function: object = None


class GlobalParams:
    def __init__(self, path_filename: str):
        self.path_filename = path_filename
        self.__loaded_params = load_config(path_filename)
        self.__new_params = None
        self.lock_new_keys = True
        self.lock_excluded = ["signal_direction", ]
        self.dont_look = ["global", "features_dict"]
        pass

    def is_sectioned(self, params_to_check) -> bool:
        check_list = []
        for k in params_to_check.keys():
            check_list += [k in self.__loaded_params.keys()]
        if np.all(check_list):
            result = True
        else:
            result = False
        return result

    def _change_params(self, params_to_change: dict):
        new_params = copy.deepcopy(self.__loaded_params)
        if params_to_change:
            if self.is_sectioned(params_to_change):
                for section_name, section_value in params_to_change.items():
                    if section_name in self.__loaded_params.keys():
                        """ Try to update section items """
                        for k_new_params, v_new_params in params_to_change[section_name].items():
                            if (k_new_params in self.__loaded_params[
                                section_name].keys()) or not self.lock_new_keys or (
                                    k_new_params in self.lock_excluded):
                                new_params[section_name].update({k_new_params: v_new_params})
            else:
                for k_new_params, v_new_params in params_to_change.items():
                    """ Check every section cos we have flat dictionary (not sectioned)"""
                    for section_name in self.__loaded_params.keys():
                        if (k_new_params in self.__loaded_params[section_name].keys()) or (
                                (k_new_params in self.lock_excluded) and section_name == 'global'):
                            new_params[section_name].update({k_new_params: v_new_params})

        new_params = prepare_calculated_global_params(new_params)
        logger.debug(
            f"{self.__class__.__name__}: Recalculated \n{new_params['global']}")
        new_params['featurizer']['features_dict'] = prepare_features_dict(params_to_check=new_params,
                                                                          global_params=new_params['global'])
        logger.debug(
            f"{self.__class__.__name__}: Prepared features. \n{new_params['featurizer']['features_dict']}")
        new_params = prepare_sections_dict(params_to_check=new_params, global_params=new_params['global'])
        logger.debug(
            f"{self.__class__.__name__}: Prepared subsection. \n{new_params['generators']['train_gen']}")
        logger.debug(
            f"{self.__class__.__name__}: Prepared subsection. \n{new_params['generators']['test_gen']}")
        logger.debug(
            f"{self.__class__.__name__}: Prepared subsection. \n{new_params['generators']['val_gen']}")
        logger.debug(
            f"{self.__class__.__name__}: Prepared subsection. \n{new_params['dbdataload']}")
        return new_params

    @property
    def params(self) -> dict:
        return self.__new_params

    @params.setter
    def params(self, params_to_change: dict):
        self.__new_params = self._change_params(params_to_change)


def prepare_sections_dict(params_to_check: any,
                          global_params: dict,
                          dont_look: list = ('global', "features_dict")
                          ) -> any:
    """

    Returns:
        any:
    """

    def prepare_k(string_):
        if isinstance(string_, str):
            if get_substring(string_) is not None:
                new_string = prepare_string(string_, global_params)
            else:
                new_string = string_
        else:
            new_string = string_
        return new_string

    new_params = copy.deepcopy(params_to_check)
    for k_section, v_section in params_to_check.items():
        if not (k_section in dont_look):
            if isinstance(v_section, dict):
                new_v_section = prepare_sections_dict(v_section, global_params)
                new_params.update({k_section: new_v_section})
            else:
                if isinstance(v_section, str):
                    new_v_section = prepare_k(v_section)
                    if new_v_section == v_section and ('e-' in v_section):
                        try:
                            # noinspection PyTypeChecker
                            new_v_section = float(v_section)
                        except:
                            pass
                    elif new_v_section == v_section and v_section == 'None':
                        # noinspection PyTypeChecker
                        new_v_section = None
                    new_params.update({k_section: new_v_section})
                elif isinstance(v_section, list):
                    new_v_section: list = []
                    for list_item in v_section:
                        if isinstance(list_item, str):
                            new_list_item = prepare_k(list_item)
                        else:
                            new_list_item = list_item
                        new_v_section.append(new_list_item)
                    new_params.update({k_section: new_v_section})
    return new_params


def prepare_features_dict(params_to_check: dict, global_params: dict):
    def prepare_k(string_):
        if isinstance(string_, str):
            if get_substring(string_) is not None:
                new_string = prepare_string(string_, global_params)
            else:
                new_string = string_
        else:
            new_string = string_
        return new_string

    new_params: dict = {}
    for k, v in params_to_check['featurizer']['features_dict'].items():
        new_k = prepare_k(k)
        new_v: list = []
        for sub_v in v:
            new_sub_v = prepare_k(sub_v)
            new_v.append(new_sub_v)
        new_params.update({new_k: new_v})
    return new_params


def get_substring(text):
    pattern = r'\$\{([^$}]+)}'
    m = regex.search(pattern, string=text)
    if m:
        found = m.group(1)
    else:
        found = None
    return found


def change_substring(substring, dictionary):
    if ('int' in substring) or ('+' in substring):
        logger.debug(
            f"{__name__}: Substring sympi '{substring}'")
        new_substring = copy.deepcopy(substring)
        for k in dictionary.keys():
            if k in new_substring:
                new_substring = new_substring.replace(k, f"{dictionary[k]}")
                if isinstance(dictionary[k], (int, float)):
                    expr = eval(str(sympify(new_substring)))
                else:
                    expr = new_substring
            else:
                expr = new_substring
        new_string = expr
    else:
        new_string = dictionary.get(substring, None)
        if new_string is None:
            logger.warning(
                f"{__name__}: Key string '{substring}' not found in dictionary or 'None'. "
                f"Updating global dictionary with this key '{substring}' with 'None'")
            new_string = None
    return new_string


def prepare_string(text: str, global_dict: dict):
    found = get_substring(text)
    if found is not None:
        new_substring = change_substring(found, global_dict)
        new_text = text.replace('${' + found + '}', f'{new_substring}')
        if isinstance(new_substring, (int, float)):
            if len(f'{new_substring}') == len(new_text):
                new_text = new_substring
        elif isinstance(new_substring, (list, dict, tuple, bool)):
            new_text = new_substring
        elif new_substring is None:
            new_text = new_substring
    else:
        new_text = text
    return new_text


def prepare_calculated_global_params(global_params: dict):
    new_params = copy.deepcopy(global_params)
    time_window_hours = global_params['global']['time_window_hours']
    interval = global_params['global']['interval']
    bins = Constants.binsizes[interval]
    timeframes_period = int((time_window_hours * 60) / bins)
    new_params['global'].update({'timeframes_period': timeframes_period})
    for k_section, v_section in global_params['global'].items():
        if v_section == 'None':
            new_params['global'].update({k_section: None})
    return new_params


def prepare_config(config_dict: dict, signal_direction: str = "plus"):
    config_dict = prepare_calculated_global_params(config_dict)
    config_dict['global'].update({"signal_direction": signal_direction})

    config_dict = prepare_features_dict(config_dict)
    return config_dict


def load_config(path_filename: str) -> dict:
    """ Load General yml config file """
    with open(path_filename, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def get_datafetcher(host=ConfigPostgreSQL.HOST,
                    database=ConfigPostgreSQL.DATABASE,
                    user=ConfigPostgreSQL.USER,
                    password=ConfigPostgreSQL.PASSWORD) -> DataFetcher:
    """ Decrypt binance api key and binance api secret """

    psg_kwargs = dict(host=host,
                      database=database,
                      user=user,
                      password=password)

    """ DataFetcher object (read-only) to get data from database for using with models """
    dummy_kwargs = dict(binance_api_key="dummy",
                        binance_api_secret="dummy")

    fetcher_obj_kwargs = dict(**psg_kwargs,
                              **dummy_kwargs)

    data_fetcher_obj = DataFetcher(**fetcher_obj_kwargs)

    return data_fetcher_obj


if __name__ == '__main__':
    import os
    from datawizard.keeper import Keeper

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('mylog.log')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    """ Start test functions """
    p = Keeper.get_x_folder_path('clearml_sunday')
    path_filename = os.path.join(p, 'config.yml')
    # params = load_config(path_filename)
    # config_dict_plus = prepare_config(params, signal_direction='plus')
    # config_dict_minus = prepare_config(params, signal_direction='minus')
    # print(config_dict_plus)
    # print(config_dict_minus)
    """ End test functions """
    """ Start test GlobalParams class """
    params_obj = GlobalParams(path_filename)
    print(params_obj.params)
    params_obj.params = {"interval": "1m", "signal_direction": "plus"}
    print(params_obj.params)
    """ End test GlobalParams class """
