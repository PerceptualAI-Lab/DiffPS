# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from scipy.io import loadmat
from os import path as osp
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from typing import Tuple, Callable
import re

from datasets.base import Image, ImageList, GalleryPerQuery, DatasetReader


class PRW(DatasetReader):
    IMAGE_NAME_REG = re.compile(r'^c([0-9]+)s([0-9]+)_([0-9]+).jpg$')
    GALLERY_FILTERS = {
        'none': (lambda query_nm, query_cam, img_nm, img_cam: img_nm != query_nm),
        'same_camera': (lambda query_nm, query_cam, img_nm, img_cam: img_nm != query_nm and img_cam == query_cam),
        'cross_camera': (lambda query_nm, query_cam, img_nm, img_cam: img_nm != query_nm and img_cam != query_cam),
    }

    @staticmethod
    def get_image_source(image_name: str) -> Tuple[int, ...]:  # return camera_id, segment_id, frame_id
        return tuple(map(int, PRW.IMAGE_NAME_REG.findall(image_name)[0]))

    @staticmethod
    def get_image_size(image_name: str) -> Tuple[int, int]:  # return width, height
        return (1920, 1080) if PRW.get_image_source(image_name)[0] < 6 else (720, 576)

    def __init__(self, root: str, unlabeled_label: int = 5555):
        assert unlabeled_label > 933
        self.root = root
        self.unlabeled_label = unlabeled_label
        self.pid2label = [  # pid2label[False] is for training set, pid2label[True] is for testing set
            {pid: label for label, pid in enumerate(loadmat(osp.join(root, nm))[key].squeeze(), 1)}
            for nm, key in [('ID_train.mat', 'ID_train'), ('ID_test.mat', 'ID_test2')]
        ]  # len(pid2label[False]) = 483, len(pid2label[True]) = 450
        self.img_rosters = [
            np.char.add(np.concatenate(loadmat(osp.join(root, file))[key].squeeze(axis=1)), '.jpg')
            for file, key in [('frame_train.mat', 'img_index_train'), ('frame_test.mat', 'img_index_test')]
        ]
        self.gallery_cams = [self.get_image_source(nm)[0] for nm in self.img_rosters[True]]
        self.query_data = pd.read_table(
            osp.join(self.root, 'query_info.txt'), sep=' ', header=None, names=['pid', 'x', 'y', 'w', 'h', 'nm']
        )
        self.query_roster = np.char.add(self.query_data['nm'].values.astype('str_'), '.jpg')
        self.query_cams = [self.get_image_source(nm)[0] for nm in self.query_roster]

    def _pids2labels(self, testing_set: bool, pids: ndarray) -> ndarray[np.int64]:
        pid2label = self.pid2label[testing_set]
        return np.fromiter((pid2label.get(pid, self.unlabeled_label) for pid in pids), dtype=np.int64)

    def _to_ps_image(self, image_name: str, boxes: ndarray, labels: ndarray) -> Image:
        boxes = boxes.astype(np.float32, copy=True)  # copy, change type
        boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
        size = self.get_image_size(image_name)
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], a_min=0.0, a_max=size[0])
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], a_min=0.0, a_max=size[1])
        boxes.setflags(write=False)
        labels.setflags(write=False)
        return Image(image_name, osp.join(self.root, 'frames', image_name), boxes, labels)

    def _make_ps_image_list(self, testing_set: bool) -> ImageList:
        result = []
        for img_nm in self.img_rosters[testing_set]:
            content = loadmat(osp.join(self.root, 'annotations', img_nm + '.mat'))
            annot = content[(content.keys() & {'box_new', 'anno_file', 'anno_previous'}).pop()]
            result.append(self._to_ps_image(img_nm, annot[:, 1:], self._pids2labels(testing_set, annot[:, 0])))
        return result

    def make_training_set(self) -> ImageList:
        return self._make_ps_image_list(False)

    def make_test_set(self, split_gallery: str = 'none') -> Tuple[ImageList, ImageList, GalleryPerQuery]:
        queries = [
            self._to_ps_image(nm + '.jpg', np.asarray([[x, y, w, h]]), self._pids2labels(True, np.asarray([pid])))
            for pid, x, y, w, h, nm in self.query_data.itertuples(index=False)
        ]
        return self._make_ps_image_list(True), queries, self.gallery_per_query(split_gallery)

    def gallery_per_query(self, split_gallery: str) -> GalleryPerQuery:
        gallery_filter = self.GALLERY_FILTERS[split_gallery]
        return [[
            img_nm for img_nm, img_cam in zip(self.img_rosters[True], self.gallery_cams)
            if gallery_filter(query_nm, query_cam, img_nm, img_cam)
        ] for query_nm, query_cam in zip(self.query_roster, self.query_cams)]


if __name__ == '__main__':
    dir_ = r'D:\DataSets\PRW'
    # print(len(loadmat(osp.join(dir_, 'ID_test.mat'))['ID_test2'].squeeze()))
    # dataset = PRW(dir_)
    # dataset.testing_set()
    # _, _, test = dataset.test_set()
    # print(test)

    # table = pd.read_table(
    #     osp.join(dir_, 'query_info.txt'),
    #     sep=' ', header=None, names=['pid', 'x', 'y', 'w', 'h', 'nm'], converters={'nm': str}
    # )
    # print(table)
    # for pid, x, y, w, h, nm in table.itertuples(index=False):
    #     print(nm)
    # print(np.char.add(table['nm'].values.astype('str_'), '.jpg'))

    test = PRW(dir_)
    test.make_training_set()
    tst, _, _ = test.make_test_set()
    print(len(tst))
    print(len(test.gallery_per_query('none')[100]))
