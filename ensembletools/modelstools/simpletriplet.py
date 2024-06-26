import numpy as np

__version__ = 0.0025


class ChainStore:
    def __init__(self):
        self.instances: list = []

    def __del__(self):
        for _ in range(len(self.instances)):
            del self.instances[-1]
        del self.instances


class SimpleTriplet:
    instances = list()

    def __init__(self,
                 ref_store_obj,
                 ):
        self.ref_store_obj = ref_store_obj
        self.ref_store_obj.instances.append(self)
        self.instances = self.ref_store_obj.instances
        self.anchor_class = None
        self.__triplet = None
        self.__triplet_data = None
        self.anchor = None
        self.pos = None
        self.neg = None

    def __del__(self):
        del self.instances
        del self.ref_store_obj

    def __call__(self, triplet_data, anchor_class):
        self.anchor_class = anchor_class
        self.__triplet_data = triplet_data
        self.anchor = self.__triplet_data[0]
        self.triplet_calc()

    def recalc_all_triplets(self):
        for obj_instance in self.ref_store_obj.instances:
            obj_instance.triplet_calc()

    def recalc_ixs_triplets(self, ixs_list):
        for ix in ixs_list:
            self.ref_store_obj.instances[ix].triplet_calc()

    def duplicate_triplet(self):
        new_st_obj = SimpleTriplet(self.ref_store_obj)
        new_st_obj(self.__triplet_data, self.anchor_class)

    def triplet_calc(self):
        triplet_is_ok = False
        while not triplet_is_ok:
            self.pos = int(np.random.choice(self.__triplet_data[1], 1))
            self.neg = int(np.random.choice(self.__triplet_data[2], 1))
            if self.neg < self.pos:
                triplet_is_ok = True
            elif (len(self.__triplet_data[1]) > 1) and (self.__triplet_data[1][-1] > self.__triplet_data[2][-1]):
                triplet_is_ok = False
            else:
                triplet_is_ok = True
        self.__triplet = (int(self.__triplet_data[0]), self.pos, self.neg)

    @property
    def triplet(self):
        return self.__triplet

    @triplet.setter
    def triplet(self, triplet_data):
        self.__triplet_data = triplet_data
        self.triplet_calc()


class SimpleDuplet:
    instances: list = []

    def __init__(self,
                 ref_store_obj,
                 ):
        self.ref_store_obj = ref_store_obj
        self.ref_store_obj.instances.append(self)
        self.instances = self.ref_store_obj.instances
        self.anchor_class = None
        self.__duplet = None
        self.__duplet_data = None
        self.anchor = None
        self.pos = None
        self.neg = None

    def __del__(self):
        del self.instances
        del self.ref_store_obj

    def __call__(self, duplet_data, anchor_class):
        self.anchor_class = anchor_class
        self.__duplet_data = duplet_data
        self.anchor = self.__duplet_data[0]
        self.duplet_calc()

    def recalc_all_duplets(self):
        for obj_instance in self.ref_store_obj.instances:
            obj_instance.duplet_calc()

    def recalc_ixs_duplets(self, ixs_list):
        for ix in ixs_list:
            self.ref_store_obj.instances[ix].duplet_calc()

    def duplicate_duplet(self):
        new_st_obj = SimpleDuplet(self.ref_store_obj)
        new_st_obj(self.__duplet_data, self.anchor_class)

    def duplet_calc(self):
        duplet_is_ok = False
        while not duplet_is_ok:
            self.pos = int(np.random.choice(self.__duplet_data[1], 1))
            self.neg = int(np.random.choice(self.__duplet_data[2], 1))
            if self.neg < self.pos:
                duplet_is_ok = True
            elif (len(self.__duplet_data[1]) > 1) and (self.__duplet_data[1][-1] > self.__duplet_data[2][-1]):
                duplet_is_ok = False
            else:
                duplet_is_ok = True
        self.__duplet = (int(self.__duplet_data[0]), self.pos), (int(self.__duplet_data[0]), self.neg)

    @property
    def duplet(self):
        return self.__duplet

    @duplet.setter
    def duplet(self, duplet_data):
        self.__duplet_data = duplet_data
        self.duplet_calc()


class BatchBalanced:
    instances: list = []

    def __init__(self,
                 ref_store_obj,
                 batch_size: int,
                 classes_qty: int,
                 ):
        self.ref_store_obj = ref_store_obj
        self.ref_store_obj.instances.append(self)
        self.instances = self.ref_store_obj.instances
        self.minor_class: int or None = None
        self.classes_qty: int = classes_qty
        self.batch_size: int = batch_size
        self.batch_class_size = self.batch_size // self.classes_qty

        if self.batch_class_size * self.classes_qty != self.batch_size:
            self.batch_class_size_left = self.batch_size - (self.batch_class_size * self.classes_qty)
        else:
            self.batch_class_size_left = 0
        self.__batch: list = []
        self.__batch_data: dict or None = None

    def __del__(self):
        del self.instances
        del self.ref_store_obj

    def __call__(self, batch_data: dict, minor_class: int):
        self.minor_class = minor_class
        self.__batch_data = batch_data
        self.minor_indices = self.__batch_data[minor_class]
        if len(self.minor_indices) < self.batch_class_size:
            self.batch_class_size = len(self.__batch)
            self.batch_size = self.batch_class_size * self.classes_qty
            self.batch_class_size_left = 0
        self.batch_calc()

    def recalc_all_batches(self):
        for obj_instance in self.ref_store_obj.instances:
            obj_instance.batch_calc()

    def recalc_ixs_batches(self, ixs_list):
        for ix in ixs_list:
            self.ref_store_obj.instances[ix].batch_calc()

    def duplicate_batches(self):
        new_st_obj = BatchBalanced(self.ref_store_obj)
        new_st_obj(self.__batch_data, self.minor_class, self.classes_qty)

    def batch_calc(self):
        """
        Calculate balanced batch

        Returns:
            None
        """
        self.__batch = list(self.__batch_data[self.minor_class])

        classes_wo_minor = list(self.__batch_data.keys())
        classes_wo_minor.remove(self.minor_class)
        rnd_class_num = self.minor_class

        if self.batch_class_size_left:
            rnd_class_num = np.random.choice(classes_wo_minor, 1)

        for class_num in classes_wo_minor:
            batch_class_size = self.batch_class_size if class_num != rnd_class_num else self.batch_class_size + self.batch_class_size_left
            self.__batch.extend(np.random.choice(self.__batch_data[class_num],
                                                 batch_class_size,
                                                 replace=False))
        self.__batch.sort()

    @property
    def batch(self):
        return self.__batch

    @batch.setter
    def batch(self, batch_data):
        """
        Save and prepare data (batch_calc) for each batch as property

        Args:
            batch_data (dict):  Batch data - dict of classes with indices
                                {class: indices }
        Returns:
            None
        """
        self.__batch_data = batch_data
        self.batch_calc()
