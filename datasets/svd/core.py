import os
import torch
import logging
import numpy as np

from collections import defaultdict


logger = logging.getLogger("datasets.svd.core")

class View(object):
    def __init__(self, cfg, mode='r'):
        self._cfg = cfg
        self._mode = mode
        if self._cfg['type'] == 'frame_level':
            self._getitem = self._getitem_frame_level
            self._setitem = self._setitem_frame_level if self._mode == 'w' else None
            self.load = self._load_frame_level_view
            self.create = self._create_frame_level_view if self._mode == 'w' else None
        elif self._cfg['type'] == 'video_level':
            self._getitem = self._getitem_video_level
            self._setitem = self._setitem_video_level if self._mode == 'w' else None
            self.load = self._load_video_level_view
            self.create = self._create_video_level_view if self._mode == 'w' else None
        else:
            raise ValueError(f"Invalid view type {self._cfg['type']}")

    def load(self):
        pass

    def create(self):
        pass

    def __getitem__(self, k):
        return self._getitem(k)

    def load_double_feat(self):
        self._max_frames = self._cfg['max_frames']
        self._q_feats, self._q_ids, self._q_indices = self._load_frame_feature(self._cfg['query']['feat_file'],
                                                                               self._cfg['query']['index_file'],
                                                                               self._cfg['query']['shape_file'],
                                                                               memmap=self._cfg['query']['memmap'])
        self._l_feats, self._l_ids, self._l_indices = self._load_frame_feature(self._cfg['labeled']['feat_file'],
                                                                               self._cfg['labeled']['index_file'],
                                                                               self._cfg['labeled']['shape_file'],
                                                                               memmap=self._cfg['labeled']['memmap'])
        self._ul_feats, self._ul_ids, self._ul_indices = self._load_frame_feature(self._cfg['unlabeled']['feat_file'],
                                                                                  self._cfg['unlabeled']['index_file'],
                                                                                  self._cfg['unlabeled']['shape_file'],
                                                                                  memmap=self._cfg['unlabeled'][
                                                                                      'memmap'])

    def _getitem_frame_level(self, k):
        if k in self._q_indices:
            offset, fn = self._q_indices[k]
            if fn <= self._max_frames:
                index = [offset+i for i in range(fn)]
            else:
                interval = (fn - 1) / (self._max_frames - 1)
                index = [offset+int(i*interval) for i in range(self._max_frames)]
            feat = torch.from_numpy(np.array(self._q_feats[index]))
        elif k in self._l_indices:
            offset, fn = self._l_indices[k]
            if fn <= self._max_frames:
                index = [offset+i for i in range(fn)]
            else:
                interval = (fn - 1) / (self._max_frames - 1)
                index = [offset+int(i*interval) for i in range(self._max_frames)]
            feat = torch.from_numpy(np.array(self._l_feats[index]))
        else:
            offset, fn = self._ul_indices[k]
            if fn <= self._max_frames:
                index = [offset+i for i in range(fn)]
            else:
                interval = (fn - 1) / (self._max_frames - 1)
                index = [offset+int(i*interval) for i in range(self._max_frames)]
            feat = torch.from_numpy(np.array(self._ul_feats[index]))

        return feat, k

    def _getitem_video_level(self, k):
        if k in self._q_indices:
            i = self._q_indices[k]
            feat = torch.from_numpy(np.array(self._q_feats[i]))
        elif k in self._l_indices:
            i = self._l_indices[k]
            feat = torch.from_numpy(np.array(self._l_feats[i]))
        else:
            i = self._ul_indices[k]
            feat = torch.from_numpy(np.array(self._ul_feats[i]))
        return feat, k

    def _load_frame_level_view(self):
        self._max_frames = self._cfg['max_frames']

        self._q_feats, self._q_ids, self._q_indices = self._load_frame_feature(self._cfg['query']['feat_file'], self._cfg['query']['index_file'], self._cfg['query']['shape_file'], memmap=self._cfg['query']['memmap'])
        self._l_feats, self._l_ids, self._l_indices = self._load_frame_feature(self._cfg['labeled']['feat_file'], self._cfg['labeled']['index_file'], self._cfg['labeled']['shape_file'], memmap=self._cfg['labeled']['memmap'])
        self._ul_feats, self._ul_ids, self._ul_indices = self._load_frame_feature(self._cfg['unlabeled']['feat_file'], self._cfg['unlabeled']['index_file'], self._cfg['unlabeled']['shape_file'], memmap=self._cfg['unlabeled']['memmap'])

    def _load_video_level_view(self):
        self._q_feats, self._q_ids, self._q_indices = self._load_video_feature(self._cfg['query']['feat_file'], self._cfg['query']['id_file'], self._cfg['query']['shape_file'], memmap=self._cfg['query']['memmap'])
        self._l_feats, self._l_ids, self._l_indices = self._load_video_feature(self._cfg['labeled']['feat_file'], self._cfg['labeled']['id_file'], self._cfg['labeled']['shape_file'], memmap=self._cfg['labeled']['memmap'])
        self._ul_feats, self._ul_ids, self._ul_indices = self._load_video_feature(self._cfg['unlabeled']['feat_file'], self._cfg['unlabeled']['id_file'], self._cfg['unlabeled']['shape_file'], memmap=self._cfg['unlabeled']['memmap'])

    def _load_frame_feature(self, feat_file, index_file, shape_file, memmap=False):
        with open(shape_file, 'r') as f:
            l = f.readline().strip().split(' ')
            h, w = tuple(map(int, l))

        if memmap:
            feat = np.memmap(feat_file, mode='r', dtype='float32', shape=(h, w))
        else:
            feat = np.load(feat_file)

        indices = dict()
        ids = list()
        with open(index_file, 'r') as f:
            for l in f:
                l = l.strip().split(' ')
                l = tuple(map(int, l))
                ids.append(l[0])
                indices[l[0]] = l[1:]

        return feat, ids, indices

    def _load_video_feature(self, feat_file, id_file, shape_file, memmap=False):
        with open(shape_file, 'r') as f:
            l = f.readline().strip().split(' ')
            h, w = tuple(map(int, l))

        if memmap:
            feat = np.memmap(feat_file, mode='r', dtype='float32', shape=(h, w))
        else:
            feat = np.load(feat_file)

        indices = dict()
        ids = list()
        with open(id_file, 'r') as f:
            for i, l in enumerate(f):
                l = int(l.strip())
                ids.append(l)
                indices[l] = i

        return feat, ids, indices

    def __setitem__(self, k, v):
        self._setitem(k, v)

    def _setitem_frame_level(self, k, v):
        if k in self._q_indices:
            i = self._q_indices[k]
            self._q_feats[i] = v
        elif k in self._l_indices:
            i = self._l_indices[k]
            self._l_feats[i] = v
        else:
            i = self._ul_indices[k]
            self._ul_feats[i] = v

    def _setitem_video_level(self, k, v):
        if k in self._q_indices:
            i = self._q_indices[k]
            self._q_feats[i] = v
        elif k in self._l_indices:
            i = self._l_indices[k]
            self._l_feats[i] = v
        else:
            i = self._ul_indices[k]
            self._ul_feats[i] = v

    def _create_frame_level_view(self, dim, query_ids, labeled_ids, unlabeled_ids):
        self._q_feats, self._q_ids, self._q_indices = \
            self._create_frame_feature(
                self._cfg['query']['feat_file'],
                self._cfg['query']['index_file'],
                self._cfg['query']['shape_file'],
                dim,
                query_ids,
                memmap=self._cfg['query']['memmap']
            )
        self._l_feats, self._l_ids, self._l_indices = \
            self._create_frame_feature(
                self._cfg['labeled']['feat_file'],
                self._cfg['labeled']['index_file'],
                self._cfg['labeled']['shape_file'],
                dim,
                labeled_ids,
                memmap=self._cfg['labeled']['memmap']
            )
        self._ul_feats, self._ul_ids, self._ul_indices = \
            self._create_frame_feature(
                self._cfg['unlabeled']['feat_file'],
                self._cfg['unlabeled']['index_file'],
                self._cfg['unlabeled']['shape_file'],
                dim,
                unlabeled_ids,
                memmap=self._cfg['unlabeled']['memmap']
            )

    def _create_video_level_view(self, dim, query_ids, labeled_ids, unlabeled_ids):
        self._q_feats, self._q_ids, self._q_indices = \
            self._create_video_feature(
                self._cfg['query']['feat_file'],
                self._cfg['query']['id_file'],
                self._cfg['query']['shape_file'],
                dim,
                query_ids,
                memmap=self._cfg['query']['memmap']
            )
        self._l_feats, self._l_ids, self._l_indices = \
            self._create_video_feature(
                self._cfg['labeled']['feat_file'],
                self._cfg['labeled']['id_file'],
                self._cfg['labeled']['shape_file'],
                dim,
                labeled_ids,
                memmap=self._cfg['labeled']['memmap']
            )
        self._ul_feats, self._ul_ids, self._ul_indices = \
            self._create_video_feature(
                self._cfg['unlabeled']['feat_file'],
                self._cfg['unlabeled']['id_file'],
                self._cfg['unlabeled']['shape_file'],
                dim,
                unlabeled_ids,
                memmap=self._cfg['unlabeled']['memmap']
            )

    def _create_frame_feature(self, feat_file, id_file, shape_file, dim, ids, memmap=False):
        #pass
        save_dir = os.path.dirname(feat_file)
        os.makedirs(save_dir, exist_ok=True)

        h, w = len(ids), dim
        with open(shape_file, 'w') as f:
            f.write(str(h)+' '+str(w))

        with open(id_file, 'w') as f:
            for i in ids:
                f.write(str(i)+'\n')

        if memmap:
            feat = np.memmap(feat_file, mode='w+', dtype='float32', shape=(h, w))
        else:
            feat = np.empty([h, w], dtype='float32')

        indices = {x: i for i, x in enumerate(ids)}

        return feat, ids, indices

    def _create_video_feature(self, feat_file, id_file, shape_file, dim, ids, memmap=False):
        save_dir = os.path.dirname(feat_file)
        os.makedirs(save_dir, exist_ok=True)

        h, w = len(ids), dim
        with open(shape_file, 'w') as f:
            f.write(str(h)+' '+str(w))

        with open(id_file, 'w') as f:
            for i in ids:
                f.write(str(i)+'\n')

        if memmap:
            feat = np.memmap(feat_file, mode='w+', dtype='float32', shape=(h, w))
        else:
            feat = np.empty([h, w], dtype='float32')

        indices = {x: i for i, x in enumerate(ids)}

        return feat, ids, indices

    def dump(self):
        if self._mode == 'r':
            return
        if not self._cfg['query']['memmap']:
            np.save(self._cfg['query']['feat_file'], self._q_feats)
        else:
            self._q_feats.flush()
        if not self._cfg['labeled']['memmap']:
            np.save(self._cfg['labeled']['feat_file'], self._l_feats)
        else:
            self._l_feats.flush()
        if not self._cfg['unlabeled']['memmap']:
            np.save(self._cfg['unlabeled']['feat_file'], self._ul_feats)
        else:
            self._ul_feats.flush()

    """
        For memmap files, the ids got these ways are more memory-access effcient.
    """
    @property
    def query_ids(self):
        return self._q_ids

    @property
    def labeled_ids(self):
        return self._l_ids

    @property
    def unlabeled_ids(self):
        return self._ul_ids


class MetaData(object):
    def __init__(self, cfg):
        self._cfg = cfg
        self._query_ids = None
        self._labeled_ids = None
        self._unlabeled_ids = None
        self._train_groundtruth = None
        self._test_groundtruth = None

    def _load_ids(self, id_file):
        ids = list()
        assert os.path.exists(id_file), f"file {id_file} does not exist!"
        with open(id_file, 'r') as f:
            for l in f:
                l = l.strip().replace('.mp4', '')
                ids.append(int(l))
        return ids

    def _load_groundtruth(self, groundtruth_file):
        gdtruth = defaultdict(dict)
        with open(groundtruth_file, 'r') as f:
            for l in f:
                l = l.strip().split(' ')
                qid = int(l[0].replace('.mp4', ''))
                cid = int(l[1].replace('.mp4', ''))
                gt = int(l[2])
                gdtruth[qid][cid] = gt
        return gdtruth

    @property
    def query_ids(self):
        if self._query_ids is None:
            self._query_ids = self._load_ids(self._cfg['query_id'])
        logger.info(f"Totally {len(self._query_ids)} query videos.")
        return self._query_ids

    @property
    def labeled_ids(self):
        if self._labeled_ids is None:
            self._labeled_ids = self._load_ids(self._cfg['labeled_id'])
        logger.info(f"Totally {len(self._labeled_ids)} labeled videos.")
        return self._labeled_ids

    @property
    def unlabeled_ids(self):
        if self._unlabeled_ids is None:
            self._unlabeled_ids = self._load_ids(self._cfg['unlabeled_id'])
        logger.info(f"Totally {len(self._unlabeled_ids)} unlabeled videos.")
        return self._unlabeled_ids

    @property
    def all_video_ids(self):
        if self._query_ids is None:
            self._query_ids = self._load_ids(self._cfg['query_id'])
        if self._labeled_ids is None:
            self._labeled_ids = self._load_ids(self._cfg['labeled_id'])
        if self._unlabeled_ids is None:
            self._unlabeled_ids = self._load_ids(self._cfg['unlabeled_id'])
        logger.info(f"Totally {len(self._query_ids)+len(self._labeled_ids)+len(self._unlabeled_ids)} videos.")
        return (self._query_ids + self._labeled_ids + self._unlabeled_ids)

    @property
    def test_groundtruth(self):
        if self._test_groundtruth is None:
            self._test_groundtruth = self._load_groundtruth(self._cfg['test_groundtruth'])
        return self._test_groundtruth

    @property
    def train_groundtruth(self):
        if self._train_groundtruth is None:
            self._train_groundtruth = self._load_groundtruth(self._cfg['train_groundtruth'])
        return self._train_groundtruth
