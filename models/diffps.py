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
from .modules.box_predictor import BoxPredictor
from .losses.oim import OIMLoss

from models.embedder import MultiGranularityEmbedding, GlobalFeatureEmbedding, Embedder
from models.sfan import SFAN
from models.dgrpn import DGPRNModulator

from utils.detection import Sampler, compute_iou, compute_centerness
from utils.general import normalize_weight_zero_bias, Pack
import sys
sys.path.append('/home/work/giyeol/generic-diffusion-feature/feature')
from diffusion_feature import FeatureExtractor
from models.aggregation_network import AggregationNetwork, AggregationNetwork2, DetectionAggregationNetwork, Frq_AggregationNetwork


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

def get_attention_map(attention_matrices):
    bs = attention_matrices.shape[0]
    normalized_attn_maps = torch.zeros_like(attention_matrices[:,1,:,1].reshape(bs,160,160))
    for i in range(bs):
        attn = attention_matrices[i,1,:,1] # first head, person token
        attn_min, attn_max = torch.min(attn), torch.max(attn)
        if attn_max - attn_min > 0:
            normalized_attn_maps[i] = ((attn - attn_min) / (attn_max - attn_min)).reshape(160,160)
    return normalized_attn_maps



class DiffPS(Module):
    def __init__(self, cfg):
        super().__init__()
        initialize = Initializer(cfg.MODEL.PARAM_INIT)
        self.cfg = cfg
        
        ''' build transform ---------------------------------------------------------------------------------------- '''
        self.transform = GeneralizedRCNNTransform(
            min_size=cfg.MODEL.TRANSFORM.MIN_SIZE,
            max_size=cfg.MODEL.TRANSFORM.MAX_SIZE,
            image_mean=cfg.MODEL.TRANSFORM.IMG_MEAN,
            image_std=cfg.MODEL.TRANSFORM.IMG_STD,
            fixed_size=cfg.FEATURE_EXTRACTOR.IMAGE_SIZE,
        )

        ''' build feature extractor -------------------------------------------------------------------------------- '''
        self.feature_map_size = cfg.FEATURE_EXTRACTOR.FEATURE_MAP_SIZE
        
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
            train_unet=cfg.FEATURE_EXTRACTOR.TRAIN_UNET,
            dtype='float32' if cfg.FEATURE_EXTRACTOR.TRAIN_UNET else 'float16'
        )
        self.t = cfg.FEATURE_EXTRACTOR.DECOUPLED_TIMESTEP

        self.decoupled_feature_extractor.offload_prompt_encoder(persistent=True)
        
        self.person_prompt_embeds = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT[0])
        self.shoes_prompt_embeds  = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT[1])
        self.pants_prompt_embeds  = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT[2])
        self.shirts_prompt_embeds = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT[3])
        self.head_prompt_embeds   = self.decoupled_feature_extractor.encode_prompt(cfg.FEATURE_EXTRACTOR.PROMPT[4])
        
        self.shoes_prompt_embeds  = self.shoes_prompt_embeds[0][0,1,:].to(cfg.DEVICE)
        self.pants_prompt_embeds  = self.pants_prompt_embeds[0][0,1,:].to(cfg.DEVICE)
        self.shirts_prompt_embeds = self.shirts_prompt_embeds[0][0,1,:].to(cfg.DEVICE)
        self.head_prompt_embeds   = self.head_prompt_embeds[0][0,1,:].to(cfg.DEVICE)
       
        text_embeds = torch.stack([
            self.head_prompt_embeds,
            self.shirts_prompt_embeds,
            self.pants_prompt_embeds,
            self.shoes_prompt_embeds
        ])
        self.register_buffer("text_embeds", text_embeds)       

        # self.reid_aggregation_network = AggregationNetwork(
        # projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
        # feature_dims=cfg.FEATURE_EXTRACTOR.REID_AGGNET_FEATURE_DIMS,
        # device=cfg.DEVICE,
        # )
        
        self.detection_aggregation_network = DetectionAggregationNetwork(
            projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
            feature_dims=cfg.FEATURE_EXTRACTOR.DETECTION_AGGNET_FEATURE_DIMS,
            device=cfg.DEVICE,
        )
        
        self.reid_frq_aggregation_network = Frq_AggregationNetwork(
            projection_dim=cfg.FEATURE_EXTRACTOR.AGGNET_OUTPUT_CHANNELS,
            feature_dims=None,
            device=cfg.DEVICE,
            frq_version=cfg.FEATURE_EXTRACTOR.FRQ_VERSION,
        ) 
        
        self.dgrpn = DGPRNModulator(tau=0.7, delta=5.0, peak_window=7, neigh_window=9, topk=50,
                     learnable_beta=True, init_beta=1.0, init_gamma=0.0)
        
        ''' build backbone head ------------------------------------------------------------------------------------ '''
        _, backbone_head_type = {
            'ResNet': (ResNetBackbone, ResNetHead),
            'ConvNeXt': (ConvNeXtBackbone, ConvNeXtHead),
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
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['feat_detection'],
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
        
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['feat_reid'],
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
    
    def forward(
            self,
            images: List[Tensor],
            targets: Optional[List[Dict[str, Tensor]]] = None,
            use_gt_as_det: bool = False
    ) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        
        original_img_sizes = [img.shape[-2:] for img in images] if not self.training else None
        images, targets = self.transform(images, targets)
        
        combined_features = self.decoupled_feature_extractor.extract(
            prompts=self.person_prompt_embeds,
            batch_size=images.tensors.shape[0],
            image=images.tensors,
            image_type='tensors',
            t=self.t
        )
      
        detection_features = {
            k: v for k, v in combined_features.items() if k in self.detection_layer_dict['layer']
        }
        det_feats = next(iter(detection_features.values()))
        
        reid_features = {
            k: v for k, v in combined_features.items() 
            if k in self.reid_layer_dict['layer'] and 'map' not in k.lower()
        }

        reid_concat_feats = []
            
        for layer_name, tensor in reid_features.items():
            resized_tensor = torch.nn.functional.interpolate(
                tensor,
                size=self.feature_map_size,
                mode='bilinear',
                align_corners=False
            )
            reid_concat_feats.append(resized_tensor)
        
        reid_concat_feats = torch.cat(reid_concat_feats, dim=1) # [bs, 2560, 160, 160]
        agg_detection_feats = self.detection_aggregation_network(det_feats.float()) # [bs, 1024, 160, 160]
        detection_attn_map = None
        detection_attn_map = get_attention_map(combined_features['up-level3-repeat0-vit-block0-cross-map'])
        dg_detection_feats = self.dgrpn(agg_detection_feats, detection_attn_map) # [bs, 1024, 160, 160]

        # agg_reid_feats = self.reid_aggregation_network(reid_concat_feats.float())
        reid_features_float32 = {k: v.float() for k, v in reid_features.items()}
        reid_features_agg = self.reid_aggregation_network(reid_features_float32) 
        reid_frq_feats = self.reid_frq_aggregation_network(reid_features_agg, training=self.training)

        detection_feats = {"feat_detection": dg_detection_feats}
        # reid_feats = {"feat_reid": agg_reid_feats} # reid_feats['feat_reid'].shape --> [bs, 1024, 160, 160]
        reid_feats = {"feat_reid": reid_frq_feats} # reid_feats['feat_reid'].shape --> [bs, 1024, 160, 160]
        import pdb; pdb.set_trace()
        
        if not use_gt_as_det:
            det_targets = [
                {'boxes': tgt['boxes'], 'labels': torch.ones_like(tgt['labels'])} for tgt in targets
            ] if targets else None
            props, rpn_losses = self.rpn(images, detection_feats, det_targets)
            boxes, scores, _, detector_losses = self.detection_roi_heads(detection_feats, props, images.image_sizes, det_targets)
        
            if self.training:
                boxes = [torch.cat([boxes_in_img, props_in_img]) for boxes_in_img, props_in_img in zip(boxes, props)]
        
        else:
            assert targets is not None 
            rpn_losses = {}
            boxes = [target['boxes'] for target in targets]
            scores = [boxes_in_img.new_ones(len(boxes_in_img)) for boxes_in_img in boxes]
            detector_losses = {}
            
        identities, reid_losses = self.reid_roi_heads(reid_feats, boxes, images.image_sizes, targets, self.text_embeds)
            
        if self.training:
            losses = rpn_losses
            losses.update(detector_losses)
            losses.update(reid_losses)
            for old_nm, new_nm in [
                ('loss_objectness', 'rpn_cls'),
                ('loss_rpn_box_reg', 'rpn_reg'),
            ]:
                if old_nm in losses:
                    losses[new_nm] = losses.pop(old_nm)
            for loss_nm in losses:
                losses[loss_nm] *= self.loss_weights.get(loss_nm, 1.0)
            return losses
        else:
            results = [
                {'boxes': boxes_in_img, 'scores': scores_in_img, 'identities': identities_in_img}
                for boxes_in_img, scores_in_img, identities_in_img in zip(boxes, scores, identities)
            ]
            results = self.transform.postprocess(results, images.image_sizes, original_img_sizes)
            return results
    
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
        self.sfan = SFAN(
            embed_dim=1024,
            num_parts=4
        )
       
    def forward(
            self,
            feats: Dict[str, Tensor],
            boxes: List[Tensor],
            image_sizes: List[Tuple[int, int]],
            targets: Optional[List[Dict[str, Tensor]]] = None,
            text_embeds: Optional[Tensor] = None
    ) -> Tuple[Optional[List[Tensor]], Optional[Dict[str, Tensor]]]:
        
        self.sfan.set_text_embeddings(text_embeds)
        
        if self.training:
            boxes, _, labels, _ = self.sampler(boxes, targets)
        else:
            labels = None

        num_boxes_per_img = list(map(len, boxes))
        box_feats = self.box_roi_pool(feats, boxes, image_sizes)
        box_feats = self.sfan(box_feats)
        box_feats = self.enhancer(box_feats)
        box_embeddings = self.embedder(box_feats)
        box_identities = box_embeddings
        box_identities = F.normalize(box_identities, dim=-1)

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


