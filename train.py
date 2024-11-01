# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from torch.utils.tensorboard import SummaryWriter
from os import path as osp
from pathlib import Path
import argparse
import yaml
from yamlinclude import YamlIncludeConstructor
import sys
import wandb
from defaults import get_default_cfg
from models.prism import PRISM
import datasets
from datasets import LoaderMaker
from engines.evaluator import Evaluator
from engines.trainer import Trainer
from utils.general import make_log_dir, set_random_seed, make_optimizer, make_scheduler

ROOT = osp.dirname(__file__)


def main(args):
    YamlIncludeConstructor.add_to_loader_class(yaml.SafeLoader)
    cfg = get_default_cfg()
    for file in args.cfg_files:
        cfg.merge_from_file(file)
    cfg.freeze()
    save_dir = str(make_log_dir(Path(ROOT), cfg.OUTPUT_DIR))
    with open(osp.join(save_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
        file.write(cfg.dump())
    set_random_seed(cfg.SEED)
    
    wandb.init(project=cfg.WANDB_PROJECT, name=cfg.WANDB_RUN, entity=cfg.WANDB_ENTITY)
    wandb.run.name = cfg.OUTPUT_DIR
    
    ''' dataset related components --------------------------------------------------------------------------------- '''
    dataset_reader = getattr(datasets, cfg.DATASET.TYPE)(cfg.DATASET.PATH)
    loader_maker = LoaderMaker(
        reader=dataset_reader,
        batch_size=cfg.DATASET.BATCH_SIZE,
    )
    training_loader = loader_maker.make_training_loader()
    test_loader, queries_loader, gallery_per_query = loader_maker.make_test_data()
    ''' training related components -------------------------------------------------------------------------------- '''
    model = PRISM(cfg).to(cfg.DEVICE)
    optimizer = make_optimizer(
        named_parameters=model.named_parameters(),
        type_=cfg.SOLVER.OPTIMIZER,
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        bias_decay=cfg.SOLVER.BIAS_DECAY,
        sgd_momentum=cfg.SOLVER.SGD_MOMENTUM,
    )
    scheduler = make_scheduler(
        optimizer=optimizer,
        iters_within_epoch=len(training_loader),
        milestones=cfg.SOLVER.LR_DECAY_MILESTONES,
        gamma=cfg.SOLVER.LR_DECAY_GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
    )
    ''' evaluator & trainer ---------------------------------------------------------------------------------------- '''
    evaluator = Evaluator(
        model=model,
        detection_iou_threshold=cfg.EVAL.DETECTION_IOU_THRESHOLD,
        search_iou_threshold=cfg.EVAL.SEARCH_IOU_THRESHOLD,
        top_k=cfg.EVAL.TOP_K,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        clip_grad=cfg.SOLVER.CLIP_GRADIENTS,
        amp=cfg.SOLVER.AMP,
    )
    ''' training --------------------------------------------------------------------------------------------------- '''
    summary_writer = SummaryWriter(save_dir)
    for _ in range(cfg.SOLVER.MAX_EPOCHS):
        for losses in trainer.train(training_loader):
            summary_writer.add_scalars('loss', losses, trainer.iteration)  # log losses
            summary_writer.add_scalars(  # log learning rate
                'lr', {f'group{index}': group['lr'] for index, group in enumerate(optimizer.param_groups)},
                trainer.iteration,
            )
        trainer.save_ckpt(osp.join(save_dir, f'ckpt_epoch-{trainer.epoch}.pth'))
        if trainer.epoch >= cfg.EVAL.START:
            kpi, vis_data = evaluator.evaluate(test_loader, queries_loader, gallery_per_query)
            summary_writer.add_scalars('detection', kpi['detection'], trainer.epoch)
            summary_writer.add_scalars('search', kpi['search'], trainer.epoch)
            for file_nm, key in [
                (f'vis_det_epoch-{trainer.epoch}.yaml', 'detection'), (f'vis_srh_epoch-{trainer.epoch}.yaml', 'search')
            ]:
                if len(vis_data[key]) > 0:  # vis_data exists
                    with open(osp.join(save_dir, file_nm), 'w', encoding='UTF-8') as file:
                        yaml.dump(vis_data[key], file)
            print(f'Epoch-{trainer.epoch}')
            yaml.dump(kpi, sys.stdout)
    summary_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", nargs='+', dest="cfg_files", help="Path to configuration file.")
    main(parser.parse_args())
