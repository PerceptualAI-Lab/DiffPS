# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from os import path as osp
from scipy.io import loadmat
import numpy as np
from numpy import ndarray
from typing import Dict, Tuple, Optional, Union, overload
from collections import namedtuple
from copy import deepcopy

from datasets.base import Image, ImageList, GalleryPerQuery, DatasetReader


class CUHK_SYSU(DatasetReader):
    """
    pool.mat:
        ['pool'].squeeze(): list[list[str * 1] * 6978]
    Images.mat:
        ['Img'].squeeze(): list[struct(imname, nAppear, box) * 18184]
        box: list[list[struct(idlocate, ishard) * N] * 1]
    Train.mat:
        ['Train'].squeeze(): list[list[list[struct(idname, nAppear, scene) * 1] * 1] * 5532]
        scene: list[list[struct(imname, idlocate, ishard) * N] * 1]
    TestGx.mat:
        ['TestGx'].squeeze(): list[struct(Query, Gallery) * 2900]
        Query: list[list[struct(imname, idlocate, ishard, idname) * 1] * 1]
        Gallery: list[list[struct(imname, idlocate, ishard) * x] * 1]
    Occlusion.mat:
        ['Occlusion1'].squeeze(): list[struct(Query, Gallery, occlusion) * 187]
        Query: list[list[struct(imname, idlocate, ishard, idname) * 1] * 1]
        Gallery: list[list[struct(imname, idlocate, ishard) * 100] * 1]
    Resolution.mat:
        ['Test_Size'].squeeze(): list[struct(Query, Gallery) * 290]
        Query: list[list[struct(imname, idlocate, ishard, idname) * 1] * 1]
        Gallery: list[list[struct(imname, idlocate, ishard) * 100] * 1]
    imname: list[str * 1]
    idlocate: list[tuple(x1, y1, w, h) * 1]
    """
    ImageAnnotation = namedtuple('ImageAnnotation', ['boxes', 'labels'])
    ImageAnnotations = Dict[str, ImageAnnotation]
    TEST_SET_CONFIG = {
        gallery_size: (f'annotation/test/train_test/TestG{gallery_size}.mat', f'TestG{gallery_size}')
        for gallery_size in [50, 100, 500, 1000, 2000, 4000]  # These test file have a same query sequence
    } | {
        'occlusion': ('annotation/test/subset/Occlusion.mat', 'Occlusion1'),
        'low_resolution': ('annotation/test/subset/Resolution.mat', 'Test_Size'),
    }

    def __init__(self, root: str, unlabeled_label: int = 5555):
        assert unlabeled_label > 5532
        self._root = root
        tst_set_img_nms = set(np.concatenate(loadmat(osp.join(root, 'annotation/pool.mat'))['pool'].squeeze()))
        images = loadmat(osp.join(root, 'annotation/Images.mat'))['Img'].squeeze()
        self._trn_set_img_annots: CUHK_SYSU.ImageAnnotations = {}
        self._tst_set_img_annots: CUHK_SYSU.ImageAnnotations = {}
        for (name,), _, (packaged_boxes,) in images:  # packaged_boxes: list[struct(idlocate, ishard) * N]
            boxes = np.concatenate(packaged_boxes['idlocate'])  # ndarray(N, 4), box_type: (x1, y1, w, h)
            ''' There are boxes with zero width or height in the dataset, which is an error generated when annotating 
                the dataset. So we must filter out these boxes. '''
            boxes = boxes[np.all(boxes >= [0, 0, 10, 20], axis=1)]  # filter
            labels = np.full(len(boxes), unlabeled_label, dtype=np.int64)
            (self._tst_set_img_annots if name in tst_set_img_nms else self._trn_set_img_annots)[name] = \
                CUHK_SYSU.ImageAnnotation(boxes, labels)

    @staticmethod
    def _set_label(image_annotations: ImageAnnotations, image_name: str, box: ndarray, label: int) -> None:
        annot = image_annotations[image_name]
        for index in range(len(annot.boxes)):
            if np.all(annot.boxes[index] == box):
                annot.labels[index] = label
                return
        assert False, 'Label annotation failed, cannot find the box in the boxes.'

    def _to_ps_image(self, image_name: str, boxes: ndarray, labels: ndarray) -> Image:
        boxes = boxes.astype(np.float32, copy=True)  # copy and change type
        boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
        boxes.setflags(write=False)
        labels.setflags(write=False)
        return Image(image_name, osp.join(self._root, 'Image/SSM', image_name), boxes, labels)

    def _to_ps_image_list(self, image_annotations: ImageAnnotations) -> ImageList:
        return [self._to_ps_image(img_nm, boxes, labels) for img_nm, (boxes, labels) in image_annotations.items()]

    def make_training_set(self) -> ImageList:
        img_annots = deepcopy(self._trn_set_img_annots)
        training_set = loadmat(osp.join(self._root, 'annotation/test/train_test/Train.mat'))['Train'].squeeze()
        for label, (((_, _, (scenes,)),),) in enumerate(training_set, 1):
            for (img_nm,), (box,), _ in scenes:
                self._set_label(img_annots, img_nm, box, label)
        return self._to_ps_image_list(img_annots)

    def _make_test_set_(self, test_set: ndarray) -> Tuple[ImageList, ImageList, GalleryPerQuery]:
        img_annots = deepcopy(self._tst_set_img_annots)
        queries, gallery_per_query = [], []
        for label, (((((query_nm,), (query_box,), _, _),),), (gallery,), *_) in enumerate(test_set, 1):
            self._set_label(img_annots, query_nm, query_box, label)
            queries.append(self._to_ps_image(query_nm, query_box[None], np.asarray([label], dtype=np.int64)))
            for (img_nm,), (box,), _ in gallery:
                if len(box) == 0:
                    break
                self._set_label(img_annots, img_nm, box, label)
            gallery_per_query.append(np.concatenate(gallery['imname']))
        return self._to_ps_image_list(img_annots), queries, gallery_per_query

    def _get_test_set_data(self, arg: Union[int, str]) -> ndarray:
        path, key = CUHK_SYSU.TEST_SET_CONFIG[arg]
        return loadmat(osp.join(self._root, path))[key].squeeze()

    @overload
    def make_test_set(self) -> Tuple[ImageList, ImageList, GalleryPerQuery]: ...

    @overload
    def make_test_set(self, gallery_size: int) -> Tuple[ImageList, ImageList, GalleryPerQuery]: ...

    @overload
    def make_test_set(self, subset_name: str) -> Tuple[ImageList, ImageList, GalleryPerQuery]: ...

    def make_test_set(self, arg: Union[int, str] = 100) -> Tuple[ImageList, ImageList, GalleryPerQuery]:
        return self._make_test_set_(self._get_test_set_data(arg))

    def gallery_per_query(self, gallery_size: int) -> GalleryPerQuery:
        return [np.concatenate(gallery['imname']) for (_, (gallery,)) in self._get_test_set_data(gallery_size)]


if __name__ == '__main__':
    dir_ = r'D:\DataSets\cuhk_sysu'

    # a = loadmat(osp.join(dir_, r'annotation\test\train_test\TestG50.mat'))['TestG50'].squeeze()['Query']
    # for i in [100, 500, 1000, 2000, 4000]:
    #     b = loadmat(osp.join(dir_, f'annotation\\test\\train_test\\TestG{i}.mat'))[f'TestG{i}'].squeeze()['Query']
    #     print(np.array_equal(a, b))

    # print(loadmat(osp.join(dir_, r'annotation\test\subset\Occlusion.mat'))['Occlusion1'].squeeze()[0])
    # print(loadmat(osp.join(dir_, r'annotation\test\subset\Resolution.mat'))['Test_Size'].squeeze().dtype)

    # a, _, _ = CUHK_SYSU(dir_).testing_set(gallery_size=50)
    # for i in [100, 500, 1000, 2000, 4000]:
    #     b, _, _ = CUHK_SYSU(dir_).testing_set(gallery_size=i)
    #     for p, q in zip(a, b):
    #         if p.name != q.name or not np.array_equal(p.boxes, q.boxes) or not np.array_equal(p.labels, q.labels):
    #             print('False')
    #             break
    #     print('End')

    test = CUHK_SYSU(dir_)
    # test.training_set()
    # test.make_test_set()
    # test.gallery_per_query(4000)
    print(test.make_test_subset('low_resolution')[0][0].labels.dtype)
