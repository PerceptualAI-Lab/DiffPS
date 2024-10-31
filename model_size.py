from os import path as osp
from pathlib import Path
import argparse
import yaml
from yamlinclude import YamlIncludeConstructor
import os
from defaults import get_default_cfg
from models.seas import SEAS
from utils.general import make_log_dir, set_random_seed
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
ROOT = osp.dirname(__file__)


def get_parameter_count(parameters):
    return sum(p.numel() for p in parameters)

def main(args):
    
    save_dir = str(make_log_dir(Path(ROOT)))
    YamlIncludeConstructor.add_to_loader_class(yaml.SafeLoader)
    cfg = get_default_cfg()
    for file in args.cfg_files:
        cfg.merge_from_file(file)
    cfg.freeze()
    with open(osp.join(save_dir, 'config.yaml'), 'w', encoding='UTF-8') as file:
        file.write(cfg.dump())
    set_random_seed(cfg.SEED)

    
    ''' training related components -------------------------------------------------------------------------------- '''
    model = SEAS(cfg).to(cfg.DEVICE)
    
    # 전체 모델의 파라미터 개수 계산
    total_params = get_parameter_count(model.parameters())
    learnable_params = get_parameter_count(p for p in model.parameters() if p.requires_grad)
    
    print(f'Total parameters: {total_params:,}')
    print(f'Learnable parameters: {learnable_params:,}')

    aggregation_network_params = get_parameter_count(model.aggregation_network.parameters())
    rpn_params = get_parameter_count(model.rpn.parameters())
    detection_roi_head_params = get_parameter_count(model.detection_roi_heads.parameters())
    reid_roi_head_params = get_parameter_count(model.reid_roi_heads.parameters())
    
    print(f'AggregationNetwork parameters: {aggregation_network_params:,}')
    print(f'RPN parameters: {rpn_params:,}')
    print(f'Detection ROIHeads parameters: {detection_roi_head_params:,}')
    print(f'ReID ROIHeads parameters: {reid_roi_head_params:,}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", nargs='+', dest="cfg_files", help="Path to configuration file.")
    main(parser.parse_args())
