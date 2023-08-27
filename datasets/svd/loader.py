import os
import math
import yaml
import random
import logging

from functools import reduce

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from .core import View, MetaData
from utils.registry import Registry


LOADERS = Registry()
logger = logging.getLogger("datasets.svd.loader")


def build_loader(cfg, only_test=False):
    for loader in cfg['loader'].keys():
        assert loader in LOADERS, f"loader {loader} is not defined"

    view_cfg_file = cfg["view"]
    view_cfg = load_config(view_cfg_file)
    view = View(view_cfg, mode="r")
    view.load()

    meta_cfg_file = cfg["meta"]
    meta_cfg = load_config(meta_cfg_file)
    meta = MetaData(meta_cfg)

    test_loader = LOADERS["test_loader"].build(view, meta, **cfg["loader"]["test_loader"])
    if only_test:
        return {"test_loader": test_loader}

    loaders = {"test_loader": test_loader}
    for loader in cfg['loader'].keys():
        if loader != "test_loader":
            loaders[loader] = LOADERS[loader].build(view, meta, **cfg["loader"][loader])

    return loaders


def load_config(config_file):
    assert os.path.exists(config_file), f"Config file {config_file} not found!"
    
    with open(config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return cfg


@LOADERS.register("baseline_loader")
class BaseLineLoader(object):
    def __init__(self, view, groups, negatives, batch_size=64, negative_batch_size=1024, num_workers=4):
        self._view = view
        self._groups = groups
        self._negatives = negatives

        # Samplers
        self._pair_sampler = PairSampler(batch_size, self._groups)
        self._negative_sampler = NegativeSampler(negative_batch_size, self._negatives)

        # DataLoaders
        self._pair_loader = DataLoader(
            self._view,
            batch_sampler=self._pair_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
            )
        self._negative_loader = DataLoader(
            self._view,
            batch_sampler=self._negative_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
            )

    def _collate_fn(self, batch):
        data, ids = zip(*batch)

        lens = [len(x) for x in data]
        padded = torch.zeros(len(data), max(lens), data[0].size(1))
        for i, x in enumerate(data):
            end = lens[i]
            padded[i, :end, :] = x
        lens = torch.tensor(lens).long()

        return padded, lens, ids

    @property
    def pair_loader(self):
        return self._pair_loader

    @property
    def negative_loader(self):
        return self._negative_loader

    @classmethod
    def build(cls, view, meta, batch_size=64, negative_batch_size=1024, num_workers=4):
        groundtruth = meta.train_groundtruth
        groups = list()
        ids = set()
        for qid, cdict in groundtruth.items():
            group = set()
            ids.add(qid)
            group.add(qid)
            for cid, isp in cdict.items():
                ids.add(cid)
                if isp:
                    group.add(cid)
            combine = []
            for idx_g in range(len(groups)):
                if len(group & groups[idx_g]) > 0:
                    combine.append(idx_g)

            for idx in combine:
                group.update(groups[idx])
            groups = [i for num, i in enumerate(groups) if num not in combine]
            groups.append(group)
        negatives = list(reduce(lambda x, y: x - y, [ids] + groups)) + meta.unlabeled_ids
        groups = list(map(list, groups))

        return cls(view, groups, negatives, batch_size=batch_size, negative_batch_size=negative_batch_size, num_workers=num_workers)

@LOADERS.register("test_loader")
class TestLoader(object):
    def __init__(self, view, test_groundtruth, batch_size=64, num_workers=4):
        self._view = view

        self.query_ids = self._view.query_ids
        self.labeled_ids = self._view.labeled_ids
        self.unlabeled_ids = self._view.unlabeled_ids
        self._test_groundtruth = test_groundtruth

        self._test_query_sampler = TestQuerySampler(batch_size, self._test_groundtruth)
        self._test_labeled_sampler = TestLabeledSampler(batch_size, self._test_groundtruth)
        self._test_unlabeled_sampler = UnLabeledSampler(batch_size, self._view.unlabeled_ids)

        self._query_loader = DataLoader(
            self._view,
            batch_sampler=self._test_query_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True
            )
        self._labeled_loader = DataLoader(
            self._view,
            batch_sampler=self._test_labeled_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
            )
        self._unlabeled_loader = DataLoader(
            self._view,
            batch_sampler=self._test_unlabeled_sampler,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=True
            )

    def _collate_fn(self, batch):
        data, ids = zip(*batch)
        lens = [len(x) for x in data]
        padded = torch.zeros(len(data), max(lens), data[0].size(1))
        for i, x in enumerate(data):
            end = lens[i]
            padded[i, :end, :] = x
        lens = torch.tensor(lens).long()

        return padded, lens, ids

    @property
    def query_loader(self):
        return self._query_loader

    @property
    def labeled_loader(self):
        return self._labeled_loader

    @property
    def unlabeled_loader(self):
        return self._unlabeled_loader

    @property
    def test_groundtruth(self):
        return self._test_groundtruth

    @classmethod
    def build(cls, view, meta, batch_size=64, num_workers=4):
        return cls(view, meta.test_groundtruth, batch_size, num_workers)


class PairSampler(Sampler):
    def __init__(self, batch_size, groups):
        self._batch_size = batch_size
        self._groups = groups

        logger.info(self)

    def __len__(self):
        return math.ceil(len(self._groups) / self._batch_size)

    def __str__(self):
        return f"| Train Pair Sampler | {len(self._groups)} groups, {sum([len(g) for g in self._groups])} samples | iters {self.__len__()} | batch size {self._batch_size}"

    def __iter__(self):
        random.shuffle(self._groups)
        for i in range(self.__len__()):
            groups = self._groups[i*self._batch_size:(i+1)*self._batch_size]
            r, p = zip(*map(lambda x: random.sample(x, 2), groups))

            yield list(r) + list(p)


class NegativeSampler(Sampler):
    def __init__(self, batch_size, negatives):
        self._batch_size =  batch_size
        self._negatives = negatives
        self._ptr = 0

        logger.info(self)

    def __len__(self):
        return math.ceil(len(self._id_list) / self._batch_size)

    def __str__(self):
        return f"| Negative Sampler | {len(self._negatives)} isolated negative samples | {self._batch_size} per batch"

    def __iter__(self):
        while True:
            x = self._negatives[self._ptr:self._ptr+self._batch_size]
            self._ptr = (self._ptr + self._batch_size) % len(self._negatives)
            yield x


class TestQuerySampler(Sampler):
    def __init__(self, batch_size, test_groundtruth):
        self._batch_size =  batch_size
        self._id_list = self._get_query_ids(test_groundtruth)

        logger.info(self)

    def _get_query_ids(self, test_groundtruth):
        return list(test_groundtruth.keys())

    def __len__(self):
        return math.ceil(len(self._id_list) / self._batch_size)

    def __str__(self):
        return f"| Test Query Sampler | {len(self._id_list)} queries | iters {self.__len__()} | batch size {self._batch_size}"

    def __iter__(self):
        for i in range(self.__len__()):
            x = self._id_list[i*self._batch_size:(i+1)*self._batch_size]
            yield x


class TestLabeledSampler(Sampler):
    def __init__(self, batch_size, test_groundtruth):
        self._batch_size =  batch_size
        self._id_list = self._get_labeled_ids(test_groundtruth)

        logger.info(self)

    def _get_labeled_ids(self, test_groundtruth):
        return list(reduce(lambda x, y: x | y, [set(x.keys()) for x in test_groundtruth.values()]))

    def __len__(self):
        return math.ceil(len(self._id_list) / self._batch_size)

    def __str__(self):
        return f"| Test Labeled Sampler | {len(self._id_list)} | labeled videos | iters {self.__len__()} | batch size {self._batch_size}"

    def __iter__(self):
        for i in range(self.__len__()):
            x = self._id_list[i*self._batch_size:(i+1)*self._batch_size]
            yield x


class UnLabeledSampler(Sampler):
    def __init__(self, batch_size, unlabeled_ids):
        self._batch_size =  batch_size
        self._id_list = unlabeled_ids

        logger.info(self)

    def __len__(self):
        return math.ceil(len(self._id_list) / self._batch_size)

    def __str__(self):
        return f"| Test UnLabeled Sampler | {len(self._id_list)} unlabeled videos | iters {self.__len__()} | batch size {self._batch_size}"

    def __iter__(self):
        for i in range(self.__len__()):
            x = self._id_list[i*self._batch_size:(i+1)*self._batch_size]
            yield x
