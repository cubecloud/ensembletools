import copy
from typing import Optional

# import numpy as np
# from numpy import ndarray, dtype

from dbbinance.fetcher.datautils import *
from ensembletools.modelstools.simpletriplet import SimpleTriplet, ChainStore, BatchBalanced
from itertools import cycle
from mlthread_tools import syncing

__version__ = 0.0030

logger = logging.getLogger()

pooltypes_dict = dict({"weighted": False,
                       "minor_balanced": False,
                       "average_balanced": True,
                       "one_horizon": False,
                       "regression": False,
                       "sequenced_horizons": True,
                       "triplets": True,
                       "batch_balanced": True,
                       "flatten_horizons": False
                       }
                      )


class PoolMeta:
    count = 0

    def __init__(self,
                 coords: np.ndarray = np.asarray(a=[], dtype=np.int32),
                 cats_targets: np.ndarray = np.asarray(a=[], dtype=np.int32),
                 targets_type: str = '',
                 pooltype: str = 'full',
                 batch_size: int = 32,
                 stride: int = 0,
                 window_length: int = 64,
                 overlap: int = 0,
                 sampling_rate: int = 1,
                 recalc_indices: bool = False,
                 name: str = None,
                 cached_indices_epoch: Union[int] or None = None):

        PoolMeta.count += 1
        self.idnum = int(PoolMeta.count)

        self.batch_size = batch_size
        self.stride = stride
        self.window_length = window_length
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.name = name

        self.__length: int = 0
        self.__indices: list = []
        self.full_length: int = 0
        self.horizons_qty: int = 0
        self.batches: int = 0

        # TODO : create indices cache for number of epoch
        self.cached_indices_epoch = cached_indices_epoch

        """  @property - trigger to calculate batches, horizons, length, etc  """
        self.__coords: np.ndarray
        self.coords = coords
        self.targets_type = targets_type

        self.classes: list = []
        self.classes_counts: list = []
        self.classes_weights: dict = {}
        self.classes_indices: dict = {}
        self.weights_order: dict = {}

        self.__recalc_indices_flag: bool = recalc_indices
        self._indices_getter_func = self._simple_indices_getter if not recalc_indices else self.set_recalc_indices_func(
            recalc_indices)

        """  @property - trigger to recalculate classes: weights, order, counts  """
        self.cats_targets: np.ndarray = self.prepare_pool_targets(targets_type=self.targets_type, targets=cats_targets)

        if self.cats_targets is not None:
            self.starting_classes_indices: dict = self._get_full_classes_indices()
            self.__recalc_classes_weights: bool = True
        else:
            self.starting_classes_indices = {0: list(range(self.__length))}
            self.__recalc_classes_weights: bool = False

        self.starting_weights_order = copy.deepcopy(self.weights_order)
        self.targets_type: str = targets_type
        self.__pooltype = pooltype

        """ Sequenced horizons """
        self.horizons_indices: list = []
        self.cycled_indices: any = cycle(self.horizons_indices)

        """ Triplets """
        self.store_obj = ChainStore()
        self.triplets_classes_indices: dict = {}
        self.triplets_minor_class_length: int = 0
        self.triplets_minor_class: int or None = None
        self.start_idx: int = 0

    def __del__(self):
        PoolMeta.count -= 1
        del self.starting_classes_indices
        del self.classes_indices
        del self.__indices
        del self.idnum
        del self.cycled_indices
        del self.triplets_classes_indices
        del self.store_obj

    @property
    def cats_targets(self) -> np.ndarray:
        return self.__cats_targets

    @cats_targets.setter
    def cats_targets(self, cats_targets):
        self.__cats_targets = cats_targets
        if self.__cats_targets is not None:
            self.recalc_classes_weights()

    @property
    def coords(self) -> np.ndarray:
        return self.__coords

    @coords.setter
    def coords(self, coords: np.ndarray):
        self.__coords = coords
        self.__length = self.__coords.shape[0]
        self.__indices = list(range(self.__length))
        self.full_length = self.__coords.shape[0]
        self.horizons_qty = np.argmax(self.__coords[:, 0]) + 1
        self.batches = calc_batches(self.length, self.batch_size, self.stride)

    @property
    def pooltype(self) -> str:
        return self.__pooltype

    @pooltype.setter
    def pooltype(self, ptype: str):
        self.__pooltype = ptype
        self.__recalc_classes_weights = True

    @property
    def length(self) -> int:
        return self.__length

    @property
    def indices(self) -> list:
        return self._indices_getter_func()

    @indices.setter
    def indices(self, indices: list):
        self.__indices = indices
        if self.__recalc_classes_weights:
            self.__length = len(self.__indices)
            self.recalc_classes_weights()
            self.__recalc_classes_weights = False
            self.batches = calc_batches(self.length, self.batch_size, self.stride)

    def _simple_indices_getter(self) -> list:
        return self.__indices

    def _sequenced_horizons_indices_getter(self) -> list:
        self.__indices = next(self.cycled_indices)
        return self.__indices

    def calc_sequence_hrzns_pool(self, recalc_indices_flag: bool = False):
        """
        Calculate sequence of horizons from 1st to last
        pooltype = "sequenced_horizons"

        Args:
            recalc_indices_flag (bool):

        Returns:
            self:
        """
        classes_indices = {weight_class[0]: list() for weight_class in self.weights_order}

        self.classes_indices = classes_indices
        horizons_indices: list = []
        for hor_ix in range(self.horizons_qty):
            _indices = np.where(self.coords[:, 0] == hor_ix)[0]
            horizons_indices.append(_indices)

        self.pooltype = "sequenced_horizons"
        self.set_recalc_indices_func(flag=recalc_indices_flag)
        self.horizons_indices = horizons_indices
        self.cycled_indices = cycle(self.horizons_indices)
        self.indices = next(self.cycled_indices)
        return self

    def _weighted_indices_getter(self) -> list:
        new_indices: list = []
        major_class = self.starting_weights_order[-1][0]

        """ Calculate minor classes indices based on minor classes """
        for (current_class, _) in self.starting_weights_order[:-1]:
            self.classes_indices[current_class] = self.starting_classes_indices[current_class]
            new_indices.extend(self.starting_classes_indices[current_class])

        major_class_indices = np.random.choice(self.starting_classes_indices[major_class],
                                               int((self.full_length - len(new_indices)) // self.horizons_qty),
                                               replace=False)

        self.classes_indices[major_class] = major_class_indices
        new_indices.extend(major_class_indices)

        new_indices.sort()
        self.__indices = new_indices
        return self.__indices

    def calc_weighted_pool(self, recalc_indices_flag: bool = False):
        """
        Calculate weighted Pool (dataclass)
        pooltype = "weighted"

        Args:
            recalc_indices_flag (bool):

        Returns:
            self:
        """

        self.pooltype = "weighted"
        self.set_recalc_indices_func(flag=recalc_indices_flag)
        self.indices = self._weighted_indices_getter()
        return self

    def _minor_balanced_indices_getter(self) -> list:
        new_indices = list()

        minor_class = self.starting_weights_order[0][0]
        minor_class_length = len(self.starting_classes_indices[minor_class])
        for (current_class, _) in self.starting_weights_order:
            class_indices = np.random.choice(self.starting_classes_indices[current_class], minor_class_length,
                                             replace=False)
            self.classes_indices[current_class] = class_indices
            new_indices.extend(class_indices)

        new_indices.sort()
        self.__indices = new_indices
        return self.__indices

    def calc_minor_balanced_pool(self, recalc_indices_flag: bool = False):
        """
        Calculate minor balanced Pool (dataclass)
        pooltype = "minor_balanced"

        Args:
            recalc_indices_flag (bool):

        Returns:
            self:
        """

        self.pooltype = "minor_balanced"
        self.set_recalc_indices_func(flag=recalc_indices_flag)
        self.indices = self._minor_balanced_indices_getter()
        return self

    def _average_balanced_indices_getter(self) -> list:
        new_indices = list()

        average_class_length = int(self.full_length // self.horizons_qty // len(self.starting_weights_order))
        for (current_class, _) in self.starting_weights_order:
            class_indices = np.random.choice(self.starting_classes_indices[current_class], int(average_class_length),
                                             replace=True)
            self.classes_indices[current_class] = class_indices
            new_indices.extend(class_indices)

        new_indices.sort()
        self.__indices = new_indices
        return self.__indices

    def calc_average_balanced_pool(self, recalc_indices_flag: bool = False):
        """
        Calculate average_balanced Pool (dataclass)
        for balancing we get a length of 1st horizon

        pooltype = "average_balanced"

        Args:
            recalc_indices_flag (bool):

        Returns:
            pool (Pool):
        """

        self.pooltype = "average_balanced"
        self.set_recalc_indices_func(flag=recalc_indices_flag)
        self.indices = self._average_balanced_indices_getter()
        return self

    def calc_one_hrzn_pool(self, recalc_indices_flag: bool = False):
        """
        Calculate one horizon (1st one) Pool (dataclass)
        pooltype = "one_horizon"

        Args:
            recalc_indices_flag (bool):

        Returns:
            self:
        """
        classes_indices = {weight_class[0]: list() for weight_class in self.weights_order}

        self.classes_indices = classes_indices

        for ix in range(self.full_length):
            if self.coords[ix, 0] == 0:
                self.classes_indices[self.cats_targets[ix]].append(ix)

        new_indices: list = []
        for class_indices in self.classes_indices.values():
            new_indices += class_indices

        new_indices.sort()

        self.pooltype = "one_horizon"
        self.set_recalc_indices_func(flag=recalc_indices_flag)
        self.indices = new_indices
        return self

    def _batch_balanced_indices_getter(self) -> list:
        new_indices = []
        self.store_obj.instances[0].recalc_all_batches()
        for ix in range(len(self.store_obj.instances)):
            batch = self.store_obj.instances[ix].batch
            new_indices.extend(copy.deepcopy(batch))

        self.__indices = new_indices
        return self.__indices

    def _triplets_indices_getter(self) -> list:
        new_indices = []
        triplets_indices = list()
        for (current_class, _) in self.weights_order:
            class_indices = np.random.choice(self.triplets_classes_indices[current_class],
                                             self.triplets_minor_class_length,
                                             replace=False)
            self.classes_indices[current_class] = class_indices
            triplets_indices.extend(class_indices)
        triplets_indices.sort()
        triplets_indices = np.array(triplets_indices) - self.start_idx
        self.store_obj.instances[0].recalc_ixs_triplets(triplets_indices)
        for ix in triplets_indices:
            triplet = self.store_obj.instances[ix].triplet
            new_indices.extend(triplet)

        self.__indices = new_indices
        return self.__indices

    @staticmethod
    def prepare_pool_targets(targets_type: str, targets: np.ndarray or List[np.ndarray]):
        """
        Prepare pool categorical targets for future calculation

        Also prepare:
        classes, classes_counts, classes_weights

        Args:
            targets_type:
            targets:

        Returns:
            None
        """
        cats_targets = None
        if targets_type == "categorical":
            cats_targets = targets.squeeze().astype(np.int32)
        elif targets_type == "ohe":
            cats_targets = np.argmax(targets, axis=1).astype(np.int32)
        elif targets_type == "regression":
            cats_targets = None
        elif targets_type == "mixed":
            cats_targets = None
        return cats_targets

    def recalc_classes_weights(self) -> None:
        self.classes, self.classes_counts, self.classes_weights = calc_classes_weights(
            y_data=self.cats_targets[self.__indices])
        self.weights_order = sorted(self.classes_weights.items(), key=lambda x: x[1])
        self.weights_order.reverse()

    def set_recalc_indices_func(self, flag: bool = False):
        self._indices_getter_func = self._simple_indices_getter
        self.__recalc_indices_flag = flag
        if flag:
            """
            indices getter for specific pool type
            each pool.indices request recalculate indices
            """
            if self.pooltype == "weighted":
                self._indices_getter_func = self._weighted_indices_getter
            elif self.pooltype == "minor_balanced":
                self._indices_getter_func = self._minor_balanced_indices_getter
            elif self.pooltype == "average_balanced":
                self._indices_getter_func = self._average_balanced_indices_getter
            elif self.pooltype == "sequenced_horizons":
                self._indices_getter_func = self._sequenced_horizons_indices_getter
            elif self.pooltype == "triplets":
                self._indices_getter_func = self._triplets_indices_getter
            elif self.pooltype == "batch_balanced":
                self._indices_getter_func = self._batch_balanced_indices_getter
            # elif self.pooltype == "flatten_horizons":
            #     self._indices_getter_func = self._simple_indices_getter

    def _get_full_classes_indices(self) -> dict:
        """ Prepare indices for each class  """
        classes_indices = dict()
        for current_class in self.classes:
            class_indices = list(np.argwhere(self.cats_targets == current_class).squeeze())
            classes_indices.update({current_class: class_indices})
        return classes_indices

    def calc_positive_indices(self, end_idx, y_anchor_class):

        y_indices_for_pos = np.array(self.classes_indices[y_anchor_class])
        y_indices_for_pos = np.extract(y_indices_for_pos < end_idx, y_indices_for_pos.copy())
        # y_indices_for_pos = np.extract(y_indices_for_pos >= self.start_index, y_indices_for_pos)

        if y_indices_for_pos.shape[0] == 0:
            y_indices_for_pos = [self.classes_indices[y_anchor_class][1]]
        return y_indices_for_pos

    def calc_negative_indices(self, end_idx, y_anchor_class):
        _temp_classes = list(self.classes)
        _temp_classes.remove(y_anchor_class)

        y_indices_for_neg = list()

        for class_num in _temp_classes:
            y_indices_for_neg += self.classes_indices[class_num]
        y_indices_for_neg.sort()
        y_indices_for_neg = np.asarray(y_indices_for_neg)
        y_indices_for_neg = np.extract(y_indices_for_neg < end_idx, y_indices_for_neg)
        # y_indices_for_neg = np.extract(y_indices_for_neg >= self.start_index, y_indices_for_neg)
        return y_indices_for_neg

    def create_pool(self, pooltype: str, recalc_indices_flag=None):
        logger.debug(
            f"{self.__class__.__name__} #{self.idnum}: creating new pool with type '{pooltype}'")

        assert pooltype in list(pooltypes_dict.keys()), f"Error: unknown pooltype '{pooltype}'"
        if self.targets_type != "regression":
            if recalc_indices_flag is None:
                new_pool = self.create_pooltype(pooltype, recalc_indices_flag=pooltypes_dict[pooltype])
            else:
                new_pool = self.create_pooltype(pooltype, recalc_indices_flag=recalc_indices_flag)
        else:
            new_pool = self
        return new_pool

    @staticmethod
    def get_classes_indices(pool) -> dict:
        classes_indices = dict()
        for w_ix in range(len(pool.weights_order)):
            current_class = pool.weights_order[w_ix][0]
            class_indices = list(np.argwhere(pool.cats_targets == current_class).squeeze())
            classes_indices.update({current_class: class_indices})
        return classes_indices

    def create_pooltype(self, ptype: str, recalc_indices_flag: bool = False):
        pooltypes_dict: dict = {"weighted": self.calc_weighted_pool,
                                "one_horizon": self.calc_one_hrzn_pool,
                                "minor_balanced": self.calc_minor_balanced_pool,
                                "average_balanced": self.calc_average_balanced_pool,
                                "sequenced_horizons": self.calc_sequence_hrzns_pool,
                                "triplets": calc_triplets_pool,
                                "batch_balanced": calc_batch_balanced_pool,
                                "flatten_horizons": calc_flatten_hrzns_pool,
                                }
        assert ptype in list(pooltypes_dict.keys()), f"Error: unknown pooltype {ptype}"
        pooltype_func = pooltypes_dict[ptype]
        if ptype in ["triplets", "batch_balanced", "flatten_horizons"]:
            with syncing.mlt_mutex:
                new_pool = pooltype_func(self, recalc_indices_flag=recalc_indices_flag)
        else:
            new_pool = pooltype_func(recalc_indices_flag=recalc_indices_flag)
        return new_pool


class Pool(PoolMeta):
    def __init__(self, coords: np.ndarray = np.asarray(a=[], dtype=np.int32),
                 cats_targets: np.ndarray = np.asarray(a=[], dtype=np.int32), targets_type: str = "",
                 pooltype: str = "full", batch_size: int = 32, stride: int = 0, window_length: int = 64,
                 overlap: int = 0, sampling_rate: int = 1, recalc_indices: bool = False, name: str = None,
                 cached_indices_epoch: Optional[int] or None = None):
        super().__init__(coords, cats_targets, targets_type, pooltype, batch_size, stride, window_length, overlap,
                         sampling_rate, recalc_indices, name, cached_indices_epoch)


def calc_batch_balanced_pool(pool: Pool, recalc_indices_flag: bool = False) -> Pool:
    """
    Calculate sequence of horizons from 1st to last
    pooltype = "batch_balanced"

    Args:
        pool (Pool):
        recalc_indices_flag (bool):

    Returns:
        pool (Pool):
    """

    new_pool = copy.deepcopy(pool)
    minor_class = new_pool.starting_weights_order[0][0]
    # minor_class_length = len(new_pool.starting_classes_indices[minor_class])

    classes_qty = len(new_pool.classes)
    class_batch_size = int(new_pool.batch_size // classes_qty)

    # first_ix = class_batch_size
    check_list = []
    class_batch_end_idx: dict = {class_num: [0, class_batch_size] for class_num in new_pool.classes}

    while True:
        start_idx = new_pool.starting_classes_indices[minor_class][class_batch_end_idx[minor_class][1]]
        for (class_num, _) in new_pool.weights_order[1:]:
            current_class_indices = np.array(new_pool.starting_classes_indices[class_num])
            current_class_indices = current_class_indices[current_class_indices < start_idx]
            indices_len = current_class_indices.shape[0]
            if indices_len > 0:
                class_batch_end_idx[class_num][1] = indices_len
                check_list.append(True)
            else:
                check_list.append(False)
        if not np.all(check_list):
            class_batch_end_idx[minor_class][1] += 1
            check_list = []
        else:
            break

    minor_class_indices = new_pool.starting_classes_indices[minor_class][class_batch_end_idx[minor_class][1]:]
    minor_class_length = len(minor_class_indices)
    batches_qty = calc_batches(data_length=minor_class_length, batch_size=class_batch_size, stride=1)

    class_batch_slices: dict = {}
    for class_num in new_pool.classes:
        current_class_batch_slice = int(
            len(new_pool.starting_classes_indices[class_num][class_batch_end_idx[class_num][1]:]) // batches_qty)
        class_batch_slices.update({class_num: current_class_batch_slice})

    for batch_ix in range(batches_qty + 1):
        base_data: dict = {}
        for class_num in new_pool.classes:
            current_start_idx = class_batch_end_idx[class_num][0]
            current_end_idx = class_batch_end_idx[class_num][1]
            current_class_indices = new_pool.starting_classes_indices[class_num]
            current_class_indices = current_class_indices[current_start_idx:current_end_idx]
            base_data.update({class_num: list(current_class_indices)})
            class_batch_end_idx.update({class_num: (
                current_start_idx + class_batch_slices[class_num], current_end_idx + class_batch_slices[class_num])})

        batch_data = copy.deepcopy(base_data)
        st = BatchBalanced(new_pool.store_obj, batch_size=new_pool.batch_size, classes_qty=classes_qty)
        st(batch_data=batch_data, minor_class=minor_class)

    new_indices = []
    for ix in range(batches_qty):
        batch = new_pool.store_obj.instances[ix].batch
        new_indices.extend(copy.deepcopy(batch))

    new_pool.pooltype = "batch_balanced"
    new_pool.set_recalc_indices_func(flag=recalc_indices_flag)
    new_pool.indices = copy.deepcopy(new_indices)
    return new_pool


def calc_triplets_pool(pool: Pool, recalc_indices_flag: bool = False) -> Pool:
    """
    Calculate triplets pool
    pooltype = "triplets"

    Args:
        pool (Pool):
        recalc_indices_flag (bool):

    Returns:
        pool (Pool):
    """
    new_pool = copy.deepcopy(pool)
    """ Get minor class 2nd index as start point """
    minor_class = new_pool.weights_order[0][0]
    minor_class_length = len(new_pool.starting_classes_indices[minor_class])
    first_ix = 1
    check_list = []

    while True or start_idx != minor_class_length:
        start_idx = new_pool.starting_classes_indices[minor_class][first_ix]
        for (class_num, _) in new_pool.weights_order[1:]:
            check_list.append((new_pool.starting_classes_indices[class_num][first_ix] < start_idx))
        if not np.all(check_list):
            first_ix += 1
            check_list = []
        else:
            break

    new_pool.triplets_classes_indices = {weight_class[0]: list() for weight_class in pool.weights_order}

    for ix in range(start_idx, new_pool.full_length):
        """ get current anchor class """
        anchor_ix = int(ix)
        anchor_class = new_pool.cats_targets[ix]
        new_pool.triplets_classes_indices[anchor_class].append(ix)
        """ using for triplets calculation all indexes less then ix """
        y_indices_for_pos = new_pool.calc_positive_indices(ix, anchor_class)
        y_indices_for_neg = new_pool.calc_negative_indices(ix, anchor_class)
        """ 
        Fill the triplets instances list  
        Remember! start_idx is [0] in SimpleTriplet.instances list
        """
        st = SimpleTriplet(new_pool.store_obj)
        st(triplet_data=(anchor_ix, y_indices_for_pos, y_indices_for_neg), anchor_class=anchor_class)

    triplets_classes_len = {key: len(value) for key, value in new_pool.triplets_classes_indices.items()}
    triplets_classes_len = (sorted(triplets_classes_len.items(), key=lambda x: x[1]))

    new_pool.triplets_minor_class = triplets_classes_len[0][0]
    new_pool.triplets_minor_class_length = triplets_classes_len[0][1]

    triplets_indices = list()
    for (current_class, _) in new_pool.weights_order:
        class_indices = np.random.choice(new_pool.triplets_classes_indices[current_class],
                                         new_pool.triplets_minor_class_length,
                                         replace=False)
        new_pool.classes_indices[current_class] = class_indices
        triplets_indices.extend(class_indices)

    triplets_indices.sort()

    new_pool.start_idx = start_idx
    new_indices = []
    for ix in triplets_indices:
        triplet = new_pool.store_obj.instances[ix - start_idx].triplet
        new_indices.extend(triplet)

    new_pool.pooltype = "triplets"
    new_pool.set_recalc_indices_func(flag=recalc_indices_flag)
    logger.info(
        f"{new_pool.__class__.__name__}: Batch size changed. Current batch size {new_pool.batch_size} * 3 = {new_pool.batch_size * 3}")
    new_pool.batch_size = new_pool.batch_size * 3
    new_pool.indices = new_indices
    return new_pool


def calc_flatten_hrzns_pool(pool: Pool, recalc_indices_flag: bool = False) -> Pool:
    """
    Calculate flatten horizons pool
    pooltype = "flatten_horizons"

    Args:
        pool (Pool):
        recalc_indices_flag (bool):

    Returns:
        pool (Pool):
    """
    classes_indices = {weight_class[0]: list() for weight_class in pool.weights_order}

    new_pool = copy.deepcopy(pool)
    new_pool.classes_indices = classes_indices
    flatten_indices = list()
    for ix in range(new_pool.coords.shape[1]):
        _indices = pool.coords[ix, 1:]
        flatten_indices.append(_indices)

    new_pool.pooltype = "flatten_horizons"
    new_pool.set_recalc_indices_func(flag=False)
    new_pool.indices = flatten_indices
    return new_pool
