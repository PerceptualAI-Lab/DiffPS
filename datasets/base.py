# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy import ndarray
from typing import List, Tuple, Dict, Optional, NamedTuple, Union
import cv2
from abc import ABC, abstractmethod
import random
import albumentations as A
from albumentations import Compose, TransformsSeqType, BboxParams


class Image(NamedTuple):
    name: str
    path: str
    boxes: ndarray[np.float32]  # pascal_voc: (x1, y1, x2, y2)
    labels: ndarray[np.int64]


ImageList = List[Image]
GalleryPerQuery = List[Union[List[str], ndarray[np.str_]]]


class PersonSearchDataset(Dataset):
    def __init__(
            self,
            image_list: ImageList,
            transforms: Optional[TransformsSeqType] = None,
    ):
        self.table = image_list
        self.transform = Compose(
            transforms=transforms,
            bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']),
        ) if transforms is not None else None

    def __getitem__(self, index: int) -> Tuple[ndarray, Dict[str, ndarray], Dict[str, str]]:
        information = self.table[index]
        image = cv2.cvtColor(cv2.imread(information.path, flags=cv2.IMREAD_COLOR), code=cv2.COLOR_BGR2RGB)
        boxes, labels = information.boxes, information.labels
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.asarray(transformed['bboxes'], dtype=np.float32)
            labels = np.asarray(transformed['labels'], dtype=np.int64)
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255  # HWC, [0,255] -> CHW, [0,1]
        return image, {'boxes': boxes, 'labels': labels}, {'name': information.name}  # image, target, note

    def __len__(self) -> int:
        return len(self.table)


def collate_fn(batch):
    """ Equivalent to 'transpose(0, 1)' """
    return tuple(zip(*batch))


def worker_init_fn(worker_id):
    """ Set random seed for each worker to ensure reproducibility. """
    worker_seed = torch.initial_seed() % 2 ** 23
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
        image_list: ImageList,
        batch_size: int,
        training: bool,
        num_works: int = 0,
        transforms: Optional[TransformsSeqType] = None,
):
    dataset = PersonSearchDataset(
        image_list=image_list,
        transforms=transforms,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_works,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=training,
        worker_init_fn=worker_init_fn,
    )


class DatasetReader(ABC):
    @abstractmethod
    def make_training_set(self) -> ImageList:
        pass

    @abstractmethod
    def make_test_set(self, *args, **kwargs) -> Tuple[ImageList, ImageList, GalleryPerQuery]:
        pass

    @abstractmethod
    def gallery_per_query(self, *args, **kwargs) -> GalleryPerQuery:
        pass


class LoaderMaker:
    def __init__(
            self,
            reader: DatasetReader,
            batch_size: int,
    ):
        self.reader = reader
        self.batch_size = batch_size

    def make_training_loader(self) -> DataLoader:
        return make_loader(
            image_list=self.reader.make_training_set(),
            batch_size=self.batch_size,
            training=True,
            num_works=self.batch_size,
            transforms=[
                A.HorizontalFlip(p=0.5),
            ],
        )

    def _make_test_loader(self, image_list: ImageList) -> DataLoader:
        return make_loader(
            image_list=image_list,
            batch_size=self.batch_size,
            training=False,
            num_works=self.batch_size,
            transforms=None,
        )

    def make_test_data(self, *args, **kwargs) -> Tuple[DataLoader, DataLoader, GalleryPerQuery]:
        test_set, queries, gallery_per_query = self.reader.make_test_set(*args, **kwargs)
        return self._make_test_loader(test_set), self._make_test_loader(queries), gallery_per_query


