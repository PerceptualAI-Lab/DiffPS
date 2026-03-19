# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader
import wandb
from typing import Tuple, List, Dict, Optional, NamedTuple
from tqdm import tqdm
import numpy as np
from numpy import ndarray
from sklearn.metrics import average_precision_score

from datasets import GalleryPerQuery
from utils.general import ndarray_to_tensor, tensor_to_ndarray


def compute_iou_matrix(boxes1: ndarray, boxes2: ndarray) -> ndarray:
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inters_wh = (
        np.minimum(boxes1[:, None, 2:], boxes2[:, 2:]) - np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    ).clip(min=0)
    inters = inters_wh[:, :, 0] * inters_wh[:, :, 1]
    unions = areas1[:, None] + areas2 - inters
    return inters / unions


class Evaluator:
    class ImageInformation(NamedTuple):
        name: str                           # the name of this image
        truths: ndarray[np.float32]         # the true boxes of this image
        labels: ndarray[np.int64]           # the label corresponding to each true box
        boxes: ndarray[np.float32]          # the boxes detected by the model in this image
        scores: ndarray[np.float32]         # the score given by the model corresponding to each detected box
        identities: ndarray[np.float32]     # the identity vector given by the model corresponding to each detected box

    ImagesInformation = Dict[str, ImageInformation]

    class QueryInformation(NamedTuple):
        name: str                           # the name of this image
        truth: ndarray[np.float32]          # the true box of the query person in this image
        label: ndarray[np.int64]            # the label of the query person
        identity: ndarray[np.float32]       # the identity vector of the query person given by the model

    QueriesInformation = List[QueryInformation]

    def __init__(
            self,
            model: Module,
            detection_iou_threshold: float = 0.5,
            search_iou_threshold: float = -1,
            top_k: Optional[List[int]] = None,
            generate_detection_visualize_data: bool = True,
            generate_search_visualize_date: bool = True,
            use_image_encoder = False
    ):
        self.model = model
        self.det_iou_thresh = detection_iou_threshold
        self.srh_iou_thresh = search_iou_threshold
        self.top_k = top_k or [1, 5, 10]
        self.det_vis, self.srh_vis = generate_detection_visualize_data, generate_search_visualize_date
        self.device = next(model.parameters()).device
        self.use_image_encoder = use_image_encoder
        
    def match_boxes(self, truths: ndarray, boxes: ndarray) -> ndarray[np.bool_]:
        """
        Returns:
            np.any(matched_matrix, axis=0): A mask of detection boxes which has matched a target box.
        """
        if len(boxes) == 0:
            return np.empty(0, dtype=np.bool_)
        if len(truths) == 0:
            return np.zeros(len(boxes), dtype=np.bool_)
        iou_matrix = compute_iou_matrix(truths, boxes)
        iou_thresh_mask = iou_matrix >= self.det_iou_thresh
        max_iou_per_row_mask = iou_matrix.argmax(axis=1)[:, None] == np.arange(iou_matrix.shape[1])
        max_iou_per_column_mask = np.arange(iou_matrix.shape[0])[:, None] == iou_matrix.argmax(axis=0)
        matched_matrix = iou_thresh_mask & max_iou_per_row_mask & max_iou_per_column_mask
        ''' keep these greater than or equal to the iou threshold. for each target box, keep only the largest iou of all
            the detection boxes. for each detection box, keep only the largest iou of all the target boxes. '''
        return np.any(matched_matrix, axis=0)

    def evaluate_detection(self, images_information: Dict[str, ImageInformation]) -> Tuple[Dict[str, float], List]:
        vis_data = []
        cnt_true_pos, cnt_predicted_pos, cnt_real_pos = 0, 0, 0
        all_y_true, all_y_score = [], []
        for img_info in tqdm(images_information.values(), desc=f"{'Evaluating detection': <20}", unit='images'):
            matched_per_box = self.match_boxes(img_info.truths, img_info.boxes)
            cnt_true_pos += np.sum(matched_per_box)
            cnt_predicted_pos += len(img_info.boxes)
            cnt_real_pos += len(img_info.truths)
            all_y_true.append(matched_per_box)
            all_y_score.append(img_info.scores)
            if self.det_vis:
                vis_data.append({
                    'name': str(img_info.name),
                    'truths': [list(map(float, truth)) for truth in img_info.truths],
                    'detection': [
                        {'box': list(map(float, box)), 'score': float(sco), 'correct': bool(cor)}
                        for box, sco, cor in zip(img_info.boxes, img_info.scores, matched_per_box)
                    ],
                })
        precision = cnt_true_pos / cnt_predicted_pos
        recall = cnt_true_pos / cnt_real_pos
        average_precision = average_precision_score(np.concatenate(all_y_true), np.concatenate(all_y_score)) * recall
        return {'AP': float(average_precision), 'precision': float(precision), 'recall': float(recall)}, vis_data

    def match_query(self, truth: ndarray, boxes: ndarray, similarity: ndarray) -> ndarray[np.bool_]:
        """ Find the box with the highest similarity in the boxes where the iou with truth exceeds the threshold. """
        matched_per_box = np.zeros((len(boxes),), dtype=np.bool_)
        if len(truth) > 0:
            if self.srh_iou_thresh > 0:
                iou_thresh = self.srh_iou_thresh
            else:
                w, h = truth[0, 2:] - truth[0, :2]
                iou_thresh = min(0.5, (w * h * 1.0) / (w + 10) * (h + 10))  # Cited from the original CUHK-SYSU paper
            passed_idxes = np.nonzero(compute_iou_matrix(truth, boxes)[0] >= iou_thresh)[0]
            if len(passed_idxes) > 0:
                matched_per_box[passed_idxes[np.argmax(similarity[passed_idxes])]] = True
        return matched_per_box

    def evaluate_search(
            self,
            images_information: Dict[str, ImageInformation],
            queries_information: List[QueryInformation],
            gallery_per_query: GalleryPerQuery,
    ) -> Tuple[Dict[str, float], List]:
        vis_data = []
        all_recall, all_average_precision, all_top_accuracies = [], [], []
        for qry_info, gallery in tqdm(
            zip(queries_information, gallery_per_query),
            desc=f"{'Evaluating search': <20}",
            total=len(queries_information),
            unit='queries',
        ):
            cnt_true_pos, cnt_real_pos = 0, 0
            all_y_true, all_y_score = [], []
            truths = []
            for img_nm in gallery:
                img_info = images_information[img_nm]
                truth_for_query = img_info.truths[np.nonzero(img_info.labels == qry_info.label)[0]]
                similarity = img_info.identities.dot(qry_info.identity[:, None]).ravel()  # cosine similarity
                similarity = (similarity + 1) / 2  # [-1, 1] -> [0, 1]
                matched_per_box = self.match_query(truth_for_query, img_info.boxes, similarity)
                cnt_true_pos += np.sum(matched_per_box)
                cnt_real_pos += len(truth_for_query)
                all_y_true.append(matched_per_box)
                all_y_score.append(similarity)
                if self.srh_vis and len(truth_for_query) > 0:
                    truths.append({'name': str(img_nm), 'box': list(map(float, truth_for_query[0]))})
            y_true, y_score = np.concatenate(all_y_true), np.concatenate(all_y_score)
            recall = cnt_true_pos / cnt_real_pos
            all_recall.append(recall)
            average_precision = average_precision_score(y_true, y_score) * recall if cnt_true_pos > 0 else 0
            all_average_precision.append(average_precision)
            descending_idx = np.argsort(y_score)[::-1]
            sorted_y_true = y_true[descending_idx]  # arrange y_true in descending order of y_score
            all_top_accuracies.append([np.any(sorted_y_true[:k]) for k in self.top_k])
            if self.srh_vis:
                y_names = np.array(gallery).repeat(list(map(len, all_y_true)))
                y_boxes = np.concatenate([images_information[img_nm].boxes for img_nm in gallery])
                top_idx = descending_idx[:max(self.top_k)]
                vis_data.append({
                    'query': {'name': str(qry_info.name), 'box': list(map(float, qry_info.truth))},
                    'truths': truths,
                    'search': [
                        {'name': str(nm), 'box': list(map(float, box)), 'score': float(sco), 'correct': bool(cor)}
                        for nm, box, sco, cor in zip(
                            y_names[top_idx], y_boxes[top_idx], y_score[top_idx], y_true[top_idx]
                        )
                    ]
                })
        kpi = {f'top-{k}': float(accuracy) for k, accuracy in zip(self.top_k, np.mean(all_top_accuracies, axis=0))}
        kpi['mAP'] = float(np.mean(all_average_precision))
        kpi['mean-recall'] = float(np.mean(all_recall))
        return kpi, vis_data

    @torch.no_grad()
    def infer_test_set(self, test_loader: DataLoader) -> ImagesInformation:
        self.model.eval()
        images_info: Evaluator.ImagesInformation = {}
        for images, targets, notes in tqdm(test_loader, desc=f"{'Inferring images of test set': <31}", unit='batches'):
            with torch.cuda.amp.autocast():
                results = tensor_to_ndarray(self.model(ndarray_to_tensor(images, self.device)))
                for result_in_img, target_in_img, note_in_img in zip(results, targets, notes):
                    images_info[note_in_img['name']] = Evaluator.ImageInformation(
                        name=note_in_img['name'],
                        truths=target_in_img['boxes'],
                        labels=target_in_img['labels'],
                        boxes=result_in_img['boxes'],
                        scores=result_in_img['scores'],
                        identities=result_in_img['identities'],
                    )
        return images_info

    @torch.no_grad()
    def infer_queries(self, queries_loader: DataLoader) -> QueriesInformation:
        self.model.eval()
        queries_info: List[Evaluator.QueryInformation] = []
        for images, targets, notes in tqdm(queries_loader, desc=f"{'Inferring images of queries': <31}", unit='batches'):
            with torch.cuda.amp.autocast():
                results = tensor_to_ndarray(self.model(
                    images=ndarray_to_tensor(images, self.device),
                    targets=ndarray_to_tensor(targets, self.device),
                    use_gt_as_det=True,
                ))
            for result_in_img, target_in_img, note_in_img in zip(results, targets, notes):
                queries_info.append(Evaluator.QueryInformation(
                    name=note_in_img['name'],
                    truth=target_in_img['boxes'][0],
                    label=target_in_img['labels'][0],  # only one ground truth box in a query image
                    identity=result_in_img['identities'][0],
                ))
        return queries_info

    def evaluate(
            self,
            test_loader: DataLoader,
            queries_loader: DataLoader,
            gallery_per_query: GalleryPerQuery,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, list]]:
        images_info = self.infer_test_set(test_loader)
        queries_info = self.infer_queries(queries_loader)
        det_kpi, det_vis_data = self.evaluate_detection(images_info)
        srh_kpi, srh_vis_data = self.evaluate_search(images_info, queries_info, gallery_per_query)
        
        wandb.log({
            "eval/detection/AP": det_kpi['AP'],
            "eval/detection/recall": det_kpi['recall'],
            "eval/search/mAP": srh_kpi['mAP'],
        })
        for k, accuracy in srh_kpi.items():
            if k.startswith('top-'):
                wandb.log({f"eval/search/{k}": accuracy})

        return {'detection': det_kpi, 'search': srh_kpi}, {'detection': det_vis_data, 'search': srh_vis_data}
