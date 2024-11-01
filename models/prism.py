import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    functional as F, Module
)
from torchvision.models.detection._utils import BoxCoder, _box_loss
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from typing import Tuple, List, Union, Dict, Optional, Any

from .backbones.resnet import ResNetBackbone, ResNetHead
from .backbones.convnext import ConvNeXtBackbone, ConvNeXtHead
from .backbones.solider import SOLIDERBackbone, SOLIDERHead
from .modules.box_predictor import BoxPredictor
from .losses.oim import OIMLoss

from models.embedder import MultiGranularityEmbedding, GlobalFeatureEmbedding, Embedder

from utils.detection import Sampler, compute_iou, compute_centerness
from utils.general import normalize_weight_zero_bias, Pack

from diffusion_feature import FeatureExtractor
from models.aggregation_network import AggregationNetwork


class Initializer:
    @staticmethod
    def initialize_v1(module: Module) -> None:
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            raise ValueError

    @staticmethod
    def initialize_v2(module: Module) -> None:
        if hasattr(module, 'load_pretrained_weights'):
            module.load_pretrained_weights()
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            module.apply(normalize_weight_zero_bias)
        else:
            raise ValueError

    def __init__(self, mode: str):
        self._initialize = getattr(Initializer, f'initialize_{mode}')

    def __call__(self, module: Module) -> None:
        self._initialize(module)


class PRISM(Module):
    def __init__(self, cfg):
        super().__init__()
        initialize = Initializer(cfg.MODEL.PARAM_INIT)
        
        ''' build transform ---------------------------------------------------------------------------------------- '''
        self.transform = GeneralizedRCNNTransform(
            min_size=cfg.MODEL.TRANSFORM.MIN_SIZE,
            max_size=cfg.MODEL.TRANSFORM.MAX_SIZE,
            image_mean=cfg.MODEL.TRANSFORM.IMG_MEAN,
            image_std=cfg.MODEL.TRANSFORM.IMG_STD,
            fixed_size=cfg.FEATURE_EXTRACTOR.IMAGE_SIZE,
        )

        ''' build feature extractor -------------------------------------------------------------------------------- '''
        self.decouple = cfg.FEATURE_EXTRACTOR.DECOUPLE
        self.feature_map_size = cfg.FEATURE_EXTRACTOR.FEATURE_MAP_SIZE
        
        if self.decouple:
            self.detection_layer_dict = dict(
                layer={layer: True for layer in cfg.FEATURE_EXTRACTOR.DETECTION_LAYER}
            )
            self.reid_layer_dict = dict(
                layer={layer: True for layer in cfg.FEATURE_EXTRACTOR.REID_LAYER}
            )
            self.combined_layer_dict = dict(
                layer={layer: True for layer in cfg.FEATURE_EXTRACTOR.DETECTION_LAYER + cfg.FEATURE_EXTRACTOR.REID_LAYER}
            )
            
            self.decoupled_feature_extractor = FeatureExtractor(
                layer=self.combined_layer_dict['layer'],
                version=cfg.FEATURE_EXTRACTOR.VERSION,
                device=cfg.DEVICE,
                attention=cfg.FEATURE_EXTRACTOR.ATTENTION,
                img_size=cfg.FEATURE_EXTRACTOR.IMAGE_SIZE[0],
            )
            self.detection_t = cfg.FEATURE_EXTRACTOR.DETECTION_TIMESTEP
            self.reid_t = cfg.FEATURE_EXTRACTOR.REID_TIMESTEP
            self.prompt_embeds = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT)
            self.decoupled_feature_extractor.offload_prompt_encoder(persistent=True)
            
        else:
            self.shared_layer_dict = dict(
                layer={layer: True for layer in cfg.FEATURE_EXTRACTOR.SHARED_LAYER}
            )
            
            self.shared_feature_extractor = FeatureExtractor(
                layer=self.shared_layer_dict['layer'],
                version=cfg.FEATURE_EXTRACTOR.VERSION,
                device=cfg.DEVICE,
                attention=cfg.FEATURE_EXTRACTOR.ATTENTION,
                img_size=cfg.FEATURE_EXTRACTOR.IMAGE_SIZE[0],
                train_unet=cfg.FEATURE_EXTRACTOR.TRAIN_UNET,
                dtype='float32' if cfg.FEATURE_EXTRACTOR.TRAIN_UNET else 'float16',
            )
            
            self.t = cfg.FEATURE_EXTRACTOR.SHARED_TIMESTEP
            self.prompt_embeds = self.shared_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT)
            self.shared_feature_extractor.offload_prompt_encoder(persistent=True)  # to save some vram
            
            if cfg.FEATURE_EXTRACTOR.PROMPT_TUNING:
                self.prompt_embeds = list(self.prompt_embeds)
                target = [0]
                if self.prompt_embeds[2] is not None:
                    target += [2]
                meta_prompts = []
                for i in target:
                    shape = [self.prompt_embeds[i].shape[j] for j in range(len(self.prompt_embeds[i].shape))]
                    # if len(shape) == 3:
                    #     shape[1] = 20
                    meta_prompt = nn.Parameter(
                        torch.randn(shape, dtype=torch.float32),
                        requires_grad=True
                    )
                    # setattr(self, f"meta_prompt{i}", meta_prompt)
                    meta_prompts.append(meta_prompt)
                    self.prompt_embeds[i] = meta_prompt 
                self.meta_prompts = torch.nn.ParameterList(meta_prompts)        


        ''' build aggregation network ----------------------------------------------------------------------------- '''
        if self.decouple:
            self.detection_aggregation_network = AggregationNetwork(
                projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
                feature_dims=cfg.FEATURE_EXTRACTOR.DETECTION_AGGNET_FEATURE_DIMS,
                device=cfg.DEVICE,
            )
            self.reid_aggregation_network = AggregationNetwork(
                projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
                feature_dims=cfg.FEATURE_EXTRACTOR.REID_AGGNET_FEATURE_DIMS,
                device=cfg.DEVICE,
            )   
        else:
            self.shared_aggregation_network = AggregationNetwork(
                projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
                feature_dims=cfg.FEATURE_EXTRACTOR.SHARED_AGGNET_FEATURE_DIMS,
                device=cfg.DEVICE,
            )
        
        
        ''' build backbone head ------------------------------------------------------------------------------------ '''
        _, backbone_head_type = {
            'ResNet': (ResNetBackbone, ResNetHead),
            'ConvNeXt': (ConvNeXtBackbone, ConvNeXtHead),
            'SOLIDER': (SOLIDERBackbone, SOLIDERHead),
        }[cfg.MODEL.BACKBONE_HEAD]


        ''' build rpn ---------------------------------------------------------------------------------------------- '''
        anchor_generator = AnchorGenerator(
            sizes=cfg.MODEL.RPN.ANCHOR_SIZE,
            aspect_ratios=cfg.MODEL.RPN.ANCHOR_RATIO,
        )
        rpn_head = RPNHead(
            in_channels=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=rpn_head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=dict(
                training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN,
                testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
            ),
            post_nms_top_n=dict(
                training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN,
                testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
            ),
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        ''' build Faster-RCNN RoI heads ---------------------------------------------------------------------------- '''
        sampler = Sampler(
            fg_thresh=cfg.MODEL.DETECTION.SAMPLE.POS_THRESH,
            bg_thresh=cfg.MODEL.DETECTION.SAMPLE.NEG_THRESH,
            batch_size_per_image=cfg.MODEL.DETECTION.SAMPLE.BATCH_SIZE,
            positive_fraction=cfg.MODEL.DETECTION.SAMPLE.POS_FRAC,
            keep_positive_fraction=True,
            bg_label=0,
            append_gt_boxes=True,
        )
        box_coder = BoxCoder(
            weights=(10.0, 10.0, 5.0, 5.0),
        )
        
        if self.decouple:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_detection'],
                output_size=cfg.MODEL.DETECTION.FEAT_MAP_SIZE,
                sampling_ratio=2,
            )
        else:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_shared'],
                output_size=cfg.MODEL.DETECTION.FEAT_MAP_SIZE,
                sampling_ratio=2,
            )
            
        box_head = backbone_head_type(
            down_sampling=True,
        )
        initialize(box_head)
        box_predictor = BoxPredictor(
            in_channels=box_head.out_channels,
            num_classes=2,
            quality=cfg.MODEL.DETECTION.QUALITY,
            batch_norm=True,
        )
        initialize(box_predictor)
        self.detection_roi_heads = FastRCNNRoIHeads(
            # modules
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            # utils
            sampler=sampler,
            box_coder=box_coder,
            # testing parameters
            score_thresh=cfg.MODEL.DETECTION.POST_PROCESS.SCORE_THRESH,
            nms_thresh=cfg.MODEL.DETECTION.POST_PROCESS.NMS_THRESH,
            detections_per_img=cfg.MODEL.DETECTION.POST_PROCESS.DETECTIONS_PER_IMAGE,
            # loss type
            box_reg_loss_type=cfg.MODEL.DETECTION.BOX_REG_LOSS_TYPE,
            quality_loss_type=cfg.MODEL.DETECTION.QUALITY_LOSS_TYPE,
        )

        ''' build background modulation network -------------------------------------------------------------------- '''
        bmn_enhancer_head = backbone_head_type(down_sampling=True)
        initialize(bmn_enhancer_head)
        bmn_enhancer = Pack(
            body=bmn_enhancer_head,
            in_feat_name=None,
            output_both_ends=True,
        )

        num_feat_map_used = {'Downsample': 1, 'OriginalAndDownsample': 2}[cfg.MODEL.REID.FEAT_MAP_USED]
        switch = cfg.MODEL.REID.EMBEDDING
        if switch == 'MGE':
            bmn_embedder = Embedder(
                in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
                in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
                dim_out=cfg.MODEL.REID.DIM_IDENTITY,
                emb_type=MultiGranularityEmbedding,
                extra_cfg=dict(
                    num_branches=cfg.MODEL.REID.EMBEDDING_MGE.NUM_BRANCHES,
                    drop_path=cfg.MODEL.REID.EMBEDDING_MGE.DROP_PATH,
                ),
            )
        elif switch == 'GFE':
            bmn_embedder = Embedder(
                in_feat_names=bmn_enhancer.out_feat_names[-num_feat_map_used:],
                in_channels_list=bmn_enhancer.out_channels_list[-num_feat_map_used:],
                dim_out=cfg.MODEL.REID.DIM_IDENTITY,
                emb_type=GlobalFeatureEmbedding,
            )
        else:
            raise ValueError
        initialize(bmn_embedder)
        
        ''' build reid roi head ------------------------------------------------------------------------------------ '''
        sampler = Sampler(
            fg_thresh=cfg.MODEL.REID.SAMPLE.POS_THRESH,
            bg_thresh=cfg.MODEL.REID.SAMPLE.NEG_THRESH,
            batch_size_per_image=cfg.MODEL.REID.SAMPLE.BATCH_SIZE,
            positive_fraction=cfg.MODEL.REID.SAMPLE.POS_FRAC,
            keep_positive_fraction=True,
            bg_label=0,
            append_gt_boxes=True,
        )
        
        if self.decouple:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_reid'],
                output_size=cfg.MODEL.REID.FEAT_MAP_SIZE,
                sampling_ratio=2,
            )   
        else:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_shared'],
                output_size=cfg.MODEL.REID.FEAT_MAP_SIZE,
                sampling_ratio=2,
            )

        reid_loss = OIMLoss(
            num_features=cfg.MODEL.REID.DIM_IDENTITY,
            num_pids=cfg.MODEL.REID.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.REID.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.REID.LOSS.MOMENTUM,
            oim_scalar=cfg.MODEL.REID.LOSS.SCALAR,
        )
        self.reid_roi_heads = ReIdRoIHeads(
            box_roi_pool=box_roi_pool,
            enhancer=bmn_enhancer,
            embedder=bmn_embedder,
            reid_loss=reid_loss,
            sampler=sampler,
        )
        
        ''' set loss weights --------------------------------------------------------------------------------------- '''
        self.loss_weights = {
            'rpn_cls':  cfg.MODEL.LOSS_WEIGHT.RPN_CLS,
            'rpn_reg':  cfg.MODEL.LOSS_WEIGHT.RPN_REG,
            'prop_cls': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_CLS,
            'prop_reg': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_REG,
            'prop_qlt': cfg.MODEL.LOSS_WEIGHT.PROPOSAL_QLT,
            'box_reid': cfg.MODEL.LOSS_WEIGHT.BOX_REID,
        }
    
    
    '''
    
    preprocess 과정
    1. dataloader를 만들 때 이미지를 [0,1]로 정규화함
    2. forward() 함수에서 self.transform(GeneralizedRCNNTransform)을 통해 
        이미지를 resize(interpolation)하고, 
        mean과 std에 맞게 정규화하고,
        size_divisible로 인해 32로 나누어떨어지도록 padding을 추가하고, 
        target도 거기에 맞게 변경됨
    3. self.feature_extractor.extract() 함수에서 지정했던 image size로 resize(interpolation)를 진행함
        + feature_extractor는 이미지가 이미 [0,1] 범위이고, mean과 std가 [0.5, 0.5, 0.5]로 정규화되어있기를 원함
        
    --> GeneralizedRCNNTransform의 fixed_size와 feature_extractor의 img_size만 동일하면 될듯
    
    '''
    
    # 만약 fixed_size를 지정해주면, image가 interpolation을 통해 resize됨
    # 그리고 size_divisible로 인해 32로 나누어떨어지도록 padding을 추가함
    # target도 fixed_size에 맞게 값이 변경됨
    
    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
            use_gt_as_det: bool = False
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        
        # get original image size for post process, and preprocess
        original_img_sizes = [img.shape[-2:] for img in images] if not self.training else None
        images, targets = self.transform(images, targets)
        
        # extract feature map
        if self.decouple:
            detection_features = self.decoupled_feature_extractor.extract(
                prompts=self.prompt_embeds,
                batch_size=images.tensors.shape[0],
                image=images.tensors,
                image_type='tensors',
                t=self.detection_t,
            )
            reid_features = self.decoupled_feature_extractor.extract(
                prompts=self.prompt_embeds,
                batch_size=images.tensors.shape[0],
                image=images.tensors,
                image_type='tensors',
                t=self.reid_t,
            )
            detection_features = {
                k: v for k, v in detection_features.items() if k in self.detection_layer_dict['layer']
            }
            reid_features = {
                k: v for k, v in reid_features.items() if k in self.reid_layer_dict['layer']
            }
        else:
            shared_features = self.shared_feature_extractor.extract(
                prompts=self.prompt_embeds,
                batch_size=images.tensors.shape[0],
                image=images.tensors,
                image_type='tensors',
                t=self.t,
            )
            
        # 모든 layer의 feature map을 resize하고 concat
        if self.decouple:
            detection_concat_feats = []
            reid_concat_feats = []
            for layer_name, tensor in detection_features.items():
                resized_tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.feature_map_size,
                    mode='bilinear',
                    align_corners=False
                )
                detection_concat_feats.append(resized_tensor)
            for layer_name, tensor in reid_features.items():
                resized_tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.feature_map_size,
                    mode='bilinear',
                    align_corners=False
                )
                reid_concat_feats.append(resized_tensor)
            detection_concat_feats = torch.cat(detection_concat_feats, dim=1)
            reid_concat_feats = torch.cat(reid_concat_feats, dim=1)
        else:
            shared_concat_feats = []
            for layer_name, tensor in shared_features.items():
                resized_tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=self.feature_map_size,
                    mode='bilinear',
                    align_corners=False
                )
                shared_concat_feats.append(resized_tensor)
            shared_concat_feats = torch.cat(shared_concat_feats, dim=1)
        
        # concat한 feature map을 aggregation network에 넣어 channel 수를 줄임
        if self.decouple:
            agg_detection_feats = self.detection_aggregation_network(detection_concat_feats.float())  
            agg_reid_feats = self.reid_aggregation_network(reid_concat_feats.float())
            detection_feats = {"feat_detection": agg_detection_feats}
            reid_feats = {"feat_reid": agg_reid_feats}
        else:
            agg_shared_feats = self.shared_aggregation_network(shared_concat_feats.float())
            shared_feats = {"feat_shared": agg_shared_feats} 

        # detection
        if not use_gt_as_det:
            det_targets = [
                {'boxes': tgt['boxes'], 'labels': torch.ones_like(tgt['labels'])} for tgt in targets
            ] if targets else None
            
            if self.decouple:
                props, rpn_losses = self.rpn(images, detection_feats, det_targets)
                boxes, scores, _, detector_losses = self.detection_roi_heads(detection_feats, props, images.image_sizes, det_targets)
            else:
                props, rpn_losses = self.rpn(images, shared_feats, det_targets)
                boxes, scores, _, detector_losses = self.detection_roi_heads(shared_feats, props, images.image_sizes, det_targets)
            
            if self.training:
                boxes = [torch.cat([boxes_in_img, props_in_img]) for boxes_in_img, props_in_img in zip(boxes, props)]
        
        else:  # using ground truth as the detection result
            assert targets is not None
            rpn_losses = {}
            boxes = [target['boxes'] for target in targets]
            scores = [boxes_in_img.new_ones(len(boxes_in_img)) for boxes_in_img in boxes]
            detector_losses = {}
            
        # re-identity
        if self.decouple:
            identities, reid_losses = self.reid_roi_heads(reid_feats, boxes, images.image_sizes, targets)
        else:
            identities, reid_losses = self.reid_roi_heads(shared_feats, boxes, images.image_sizes, targets)
            
        # post process
        if self.training:  # wrap losses
            losses = rpn_losses
            losses.update(detector_losses)
            losses.update(reid_losses)
            for old_nm, new_nm in [  # change name
                ('loss_objectness', 'rpn_cls'),
                ('loss_rpn_box_reg', 'rpn_reg'),
            ]:
                if old_nm in losses:
                    losses[new_nm] = losses.pop(old_nm)
            for loss_nm in losses:  # multiply weight
                losses[loss_nm] *= self.loss_weights.get(loss_nm, 1.0)
            return losses
        else:  # wrap result
            results = [
                {'boxes': boxes_in_img, 'scores': scores_in_img, 'identities': identities_in_img}
                for boxes_in_img, scores_in_img, identities_in_img in zip(boxes, scores, identities)
            ]
            results = self.transform.postprocess(results, images.image_sizes, original_img_sizes)
            return results


class FastRCNNRoIHeads(Module):
    def __init__(
            self,
            box_roi_pool: MultiScaleRoIAlign,
            box_head: Module,
            box_predictor: Module,
            # utils
            box_coder: BoxCoder,
            sampler: Sampler,
            # evaluating parameters
            score_thresh: float,
            nms_thresh: float,
            detections_per_img: int,
            # loss type
            box_reg_loss_type: str = 'smooth_l1',
            quality_loss_type: str = 'iou'
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_coder = box_coder
        self.sampler = sampler
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.box_reg_loss_type = box_reg_loss_type
        if quality_loss_type == 'centerness':
            self.quality_encoder = compute_centerness
        elif quality_loss_type == 'iou':
            self.quality_encoder = compute_iou
        else:
            assert False

    def compute_losses(
            self, prop_clss: Tensor, prop_regs: Tensor, prop_qlts: Optional[Tensor],
            props: List[Tensor], truths: List[Tensor], labels: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        props, truths, labels = torch.cat(props), torch.cat(truths), torch.cat(labels)
        loss_cls = F.cross_entropy(
            input=prop_clss,
            target=labels,
        )
        fg_idxs = torch.nonzero(labels > 0).view(-1)
        loss_reg = _box_loss(
            type=self.box_reg_loss_type,
            box_coder=self.box_coder,
            anchors_per_image=props[fg_idxs],
            matched_gt_boxes_per_image=truths[fg_idxs],
            bbox_regression_per_image=prop_regs[fg_idxs],
            cnf=dict(beta=1),
        ) / labels.numel()
        loss_qlt = F.binary_cross_entropy_with_logits(
            input=prop_qlts[fg_idxs].view(-1),
            target=self.quality_encoder(truths[fg_idxs], prop_regs[fg_idxs]),
            reduction='sum',
        ) / labels.numel() if prop_qlts is not None else None
        return loss_cls, loss_reg, loss_qlt

    @torch.no_grad()
    def postprocess(
            self, prop_clss: Tensor, prop_regs: Tensor, prop_qlts: Optional[Tensor],
            props: List[Tensor], image_size: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        num_imgs, num_classes = len(props), prop_clss.shape[-1]
        num_props_per_img = list(map(len, props))
        boxes, scores, classes = [], [], []
        for prop_clss_in_img, prop_regs_in_img, prop_qlts_in_img, props_in_img, img_sz in zip(
            prop_clss.split(num_props_per_img),
            prop_regs.split(num_props_per_img),
            prop_qlts.split(num_props_per_img) if prop_qlts is not None else [None] * num_imgs,
            props, image_size
        ):
            boxes_in_img = self.box_coder.decode_single(prop_regs_in_img, props_in_img)
            boxes_in_img = box_ops.clip_boxes_to_image(boxes_in_img, img_sz)  # clip the box so that it is all inside
            keep = box_ops.remove_small_boxes(boxes_in_img, min_size=1e-2)   # remove the small boxes
            boxes_in_img = boxes_in_img[keep]
            if not self.training:  # inference
                prop_clss_in_img = prop_clss_in_img[keep]
                prop_qlts_in_img = prop_qlts_in_img[keep] if prop_qlts_in_img is not None else None

                raw_scores_in_img = F.softmax(prop_clss_in_img, dim=-1)[:, 1:]  # remove background
                retained_score_idxs = torch.nonzero(raw_scores_in_img.view(-1) > self.score_thresh).view(-1)

                scores_in_img = (F.sigmoid(prop_qlts_in_img) if prop_qlts_in_img is not None else 1) * raw_scores_in_img
                scores_in_img = scores_in_img.view(-1)[retained_score_idxs]

                boxes_in_img = boxes_in_img[retained_score_idxs // (num_classes - 1)]  # do not have background
                classes_in_img = (retained_score_idxs % (num_classes - 1)) + 1  # 0 is background

                keep = box_ops.batched_nms(boxes_in_img, scores_in_img, classes_in_img, self.nms_thresh)
                keep = keep[:self.detections_per_img]

                boxes_in_img = boxes_in_img[keep]
                scores.append(scores_in_img[keep])
                classes.append(classes_in_img[keep])
            boxes.append(boxes_in_img)
        return boxes, scores, classes

    def forward(
            self,
            feats: Dict[str, Tensor],
            props: List[Tensor],
            image_sizes: List[Tuple[int, int]],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], Optional[Dict[str, Tensor]]]:
        if self.training:
            props, truths, labels, _ = self.sampler(props, targets)
        else:
            truths, labels = None, None

        prop_feats = self.box_roi_pool(feats, props, image_sizes)
        prop_feats = self.box_head(prop_feats)
        prop_clss, prop_regs, prop_qlts = self.box_predictor(prop_feats)

        if self.training:
            loss_cls, loss_reg, loss_qlt = self.compute_losses(prop_clss, prop_regs, prop_qlts, props, truths, labels)
            losses = {'prop_cls': loss_cls, 'prop_reg': loss_reg}
            if loss_qlt is not None:
                losses['prop_qlt'] = loss_qlt
        else:
            losses = None

        boxes, scores, classes = self.postprocess(
            prop_clss.detach(), prop_regs.detach(), prop_qlts.detach() if prop_qlts is not None else None,
            props, image_sizes
        )

        return boxes, scores, classes, losses


class ReIdRoIHeads(Module):
    def __init__(
            self,
            box_roi_pool: MultiScaleRoIAlign,
            enhancer: Module,
            embedder: Module,
            reid_loss: Module,
            sampler: Sampler,
    ):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.enhancer = enhancer
        self.embedder = embedder
        self.reid_loss = reid_loss
        self.sampler = sampler

    def forward(
            self,
            feats: Dict[str, Tensor],
            boxes: List[Tensor],
            image_sizes: List[Tuple[int, int]],
            targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Optional[List[Tensor]], Optional[Dict[str, Tensor]]]:

        if self.training:
            boxes, _, labels, _ = self.sampler(boxes, targets)
        else:
            labels = None

        num_boxes_per_img = list(map(len, boxes))
        box_feats = self.box_roi_pool(feats, boxes, image_sizes)
        box_feats = self.enhancer(box_feats)
        # box_feats['head_input'].shape --> [bbox 개수, 512, 24, 12]
        # box_feats['head_output'].shape --> [bbox 개수, 512, 12, 6]
        box_embeddings = self.embedder(box_feats) # 512 embedding + 512 embedding --> 1024 embedding
        # box_embeddings.shape --> [bbox 개수, 1024]
        box_identities = box_embeddings
        box_identities = F.normalize(box_identities, dim=-1)
        # box_identities.shape --> [bbox 개수, 1024]

        if self.training:
            loss_box_reid = self.reid_loss(box_identities, labels)
            losses = {}
            if loss_box_reid is not None:
                losses['box_reid'] = loss_box_reid
            box_identities = None
        else:
            box_identities = box_identities.split(num_boxes_per_img)
            losses = None

        return box_identities, losses
