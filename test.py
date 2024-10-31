# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import yaml
import sys
import argparse
from yamlinclude import YamlIncludeConstructor

from defaults import get_default_cfg
from models.seas import SEAS
import datasets
from datasets import LoaderMaker
from utils.general import set_random_seed
from engines.evaluator import Evaluator
from engines.trainer import Trainer


def main(args):
    YamlIncludeConstructor.add_to_loader_class(yaml.SafeLoader)
    cfg = get_default_cfg()
    cfg.merge_from_file(args.cfg_file)
    cfg.freeze()
    set_random_seed(cfg.SEED)
    model = SEAS(cfg).to(cfg.DEVICE)
    model.load_state_dict(Trainer.get_model_state_dict_from_ckpt(args.ckpt_file))
    evaluator = Evaluator(
        model=model,
        detection_iou_threshold=cfg.EVAL.DETECTION_IOU_THRESHOLD,
        search_iou_threshold=cfg.EVAL.SEARCH_IOU_THRESHOLD,
        top_k=cfg.EVAL.TOP_K,
        generate_detection_visualize_data=False,
        generate_search_visualize_date=False,
    )
    dataset_reader = getattr(datasets, cfg.DATASET.TYPE)(cfg.DATASET.PATH)
    loader_maker = LoaderMaker(
        reader=dataset_reader,
        batch_size=cfg.DATASET.BATCH_SIZE,
    )
    test_loader, queries_loader, _ = loader_maker.make_test_data()
    print('Inferring:')
    images_info = evaluator.infer_test_set(test_loader)
    queries_info = evaluator.infer_queries(queries_loader)
    print('Detection:')
    det_kpi, _ = evaluator.evaluate_detection(images_info)
    yaml.dump(det_kpi, sys.stdout)
    print('Search:')
    param_nm, params = {
        'CUHK_SYSU': ('Gallery Size', [50, 100, 500, 1000, 2000, 4000]),
        'PRW': ('Split Gallery', ['none', 'cross_camera']),
    }[cfg.DATASET.TYPE]
    for param in params:
        print(f'{param_nm}: {param}')
        srh_kpi, _ = evaluator.evaluate_search(images_info, queries_info, dataset_reader.gallery_per_query(param))
        yaml.dump(srh_kpi, sys.stdout)
    if cfg.DATASET.TYPE == 'CUHK_SYSU':
        print('CUHK-SYSU subset:')
        for param in ['occlusion', 'low_resolution']:
            print(f'{param}:')
            test_loader, queries_loader, gallery_per_query = loader_maker.make_test_data(param)
            kpi, _ = evaluator.evaluate(test_loader, queries_loader, gallery_per_query)
            yaml.dump(kpi, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a person search network.')
    parser.add_argument('--cfg', dest='cfg_file', help='Path to configuration file.')
    parser.add_argument('--ckpt', dest='ckpt_file', help='Path to checkpoint file.')
    main(parser.parse_args())
