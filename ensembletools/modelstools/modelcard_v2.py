import marshmallow.validate
from ensembletools.modelstools.pooltypes import pooltypes_dict
from dbbinance.fetcher.constants import Constants
from dataclasses import dataclass, field

__version__ = 0.028


@dataclass
class ModelCard:
    model_uuid: str = field(default=str())
    model_name: str = field(default=str())
    model_type: str = field(default="classification",
                            metadata={"validate": marshmallow.validate.OneOf(["classification", "regression",
                                                                              "mixed"])})
    model_activator_key: str = field(default=str())
    model_activator_value: str = field(default=str())

    symbol: str = field(default='BTCUSDT')
    market: str = field(default='spot',
                        metadata={'validate': marshmallow.validate.OneOf(['spot', 'futures'])})
    interval: str = field(default='1h',
                          metadata={'validate': marshmallow.validate.OneOf(Constants.time_intervals)})
    power_trend: float = field(default=0.01,
                               metadata={"validate": marshmallow.validate.Range(min=0.01, min_inclusive=True)})
    time_window_hours: int = field(default=1,
                                   metadata={"validate": marshmallow.validate.Range(min=1, min_inclusive=True)})
    timeframes_period: int = field(default=1,
                                   metadata={"validate": marshmallow.validate.Range(min=1, min_inclusive=True)})
    target_steps: int = field(default=1, metadata={"validate": marshmallow.validate.Range(min=1, min_inclusive=True)})
    use_pooltype: str = field(default="weighted",
                              metadata={"validate": marshmallow.validate.OneOf(list(pooltypes_dict.keys()))}
                              )
    start_period: str = field(default=str())
    end_period: str = field(default=str())
    fixed_end_period: str = field(default=str())
