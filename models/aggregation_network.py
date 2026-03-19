import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from pytorch_wavelets import DWTForward, DWTInverse, DTCWTForward, DTCWTInverse # (or import DWT, IDWT)


class LPM(nn.Module):
    """Lightweight Processing Module: 1x1 conv + depthwise conv + 1x1 conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dw_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.dw_conv(x)))
        x = self.bn3(self.conv2(x))
        return x

class SRB(nn.Module):
    """Sub-band Refinement Block with channel-wise attention"""
    def __init__(self, channels):
        super().__init__()
        # Depthwise separable convolution
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel-wise attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Depthwise separable conv
        out = self.relu(self.bn(self.pw_conv(self.dw_conv(x))))
        
        # Channel-wise attention
        att = self.sigmoid(self.attention(self.gap(out)))
        out = out * att
        
        return out

class FRM(nn.Module):
    """Frequency Refinement Module with DWT + SRB + IDWT (논문 Eq.5: learnable γ_LH, γ_HL, γ_HH)"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.srb_lh = SRB(channels)
        self.srb_hl = SRB(channels)
        self.srb_hh = SRB(channels)
        # 논문: F_r'' = IDWT(F_LL, γ_LH F_hat_LH, γ_HL F_hat_HL, γ_HH F_hat_HH)
        self.gamma_lh = nn.Parameter(torch.tensor(1.0))
        self.gamma_hl = nn.Parameter(torch.tensor(1.0))
        self.gamma_hh = nn.Parameter(torch.tensor(1.0))

        self.DWT = DWTForward(J=1, wave='haar', mode='zero')
        self.IDWT = DWTInverse(wave='haar', mode='zero')

    def forward(self, x, training):
        if training:
            self.DWT = self.DWT.to(torch.float16)
            self.IDWT = self.IDWT.to(torch.float16)
        else:
            self.DWT = self.DWT.to(torch.float32)
            self.IDWT = self.IDWT.to(torch.float32)

        # DWT decomposition
        Yl, Yh = self.DWT(x)  # Yl: low freq, Yh[0]: high freq components

        LL = Yl

        LH = self.srb_lh(Yh[0][:, :, 0, :, :])
        HL = self.srb_hl(Yh[0][:, :, 1, :, :])
        HH = self.srb_hh(Yh[0][:, :, 2, :, :])

        LH = (self.gamma_lh * LH).unsqueeze(2)
        HL = (self.gamma_hl * HL).unsqueeze(2)
        HH = (self.gamma_hh * HH).unsqueeze(2)
        High = torch.cat([LH, HL, HH], dim=2)

        out = self.IDWT((LL, [High]))
        return out + x 

class Frq_AggregationNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims=None,
            device='cuda', 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1,
            frq_version="v1",
        ):
        super().__init__()
        self.feature_dims = feature_dims  
        self.device = device
        self.projection_dim = projection_dim
        
        # LPM modules for different input channels
        self.lpm_1280 = LPM(1280, projection_dim)  # for level1
        self.lpm_640 = LPM(640, projection_dim)    # for level2
        self.lpm_320 = LPM(320, projection_dim)    # for level3
        
        # 1x1 convs for channel adjustment after concat
        self.conv_level2 = nn.Conv2d(projection_dim * 2, projection_dim, kernel_size=1, bias=False)  # level2 features only (2개)
        self.conv_level3 = nn.Conv2d(projection_dim * 4, projection_dim, kernel_size=1, bias=False)  # level3 features only (4개)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(projection_dim)
        
        # FRM modules  
        self.frm_level2 = FRM(projection_dim)
        self.frm_level3 = FRM(projection_dim)
        
        # Final output layer
        self.final_conv = nn.Conv2d(projection_dim, projection_dim, kernel_size=1, bias=False)
        
        
        
    def forward(self, features, training):
        
        # ===== LEVEL 1 Processing (40x40 -> 80x80) =====
        level1_feat = features['up-level1-repeat2-vit-block0-self-q']  # [B, 1280, 40, 40]
        level1_out = self.lpm_1280(level1_feat)  # [B, projection_dim, 40, 40]
        
        # Upsample to 80x80
        level1_up = F.interpolate(level1_out, size=(80, 80), mode='bilinear', align_corners=False)  # [B, projection_dim, 80, 80]
        
        # ===== LEVEL 2 Processing (80x80) =====
        level2_feat1 = features['up-level2-repeat1-vit-block0-self-v']  # [B, 640, 80, 80]
        level2_feat2 = features['up-level2-repeat2-vit-block0-self-q']  # [B, 640, 80, 80]
        
        # Apply LPM to each feature
        level2_out1 = self.lpm_640(level2_feat1)  # [B, projection_dim, 80, 80]
        level2_out2 = self.lpm_640(level2_feat2)  # [B, projection_dim, 80, 80]
        
        # Concat level2 features only and apply 1x1 conv
        level2_concat = torch.cat([level2_out1, level2_out2], dim=1)  # [B, projection_dim*2, 80, 80]
        level2_processed = self.conv_level2(level2_concat)  # [B, projection_dim, 80, 80]
        
        # Element-wise addition with upsampled level1
        level2_fused = level2_processed + level1_up  # [B, projection_dim, 80, 80]
        
        # Layer normalization
        B, C, H, W = level2_fused.shape
        level2_norm = level2_fused.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        level2_norm = self.layer_norm(level2_norm)  # [B, H, W, C]
        level2_norm = level2_norm.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        # Apply FRM
        level2_refined = self.frm_level2(level2_norm, training)  # [B, projection_dim, 80, 80]
        
        # ===== LEVEL 3 Processing (160x160) =====
        level3_feat1 = features['up-level3-repeat0-vit-block0-self-q']  # [B, 320, 160, 160]
        level3_feat2 = features['up-level3-repeat0-vit-block0-self-k']  # [B, 320, 160, 160]
        level3_feat3 = features['up-level3-repeat0-vit-block0-self-v']  # [B, 320, 160, 160]
        level3_feat4 = features['up-level3-repeat1-vit-block0-self-k']  # [B, 320, 160, 160]
        
        # Apply LPM to each feature
        level3_out1 = self.lpm_320(level3_feat1)  # [B, projection_dim, 160, 160]
        level3_out2 = self.lpm_320(level3_feat2)  # [B, projection_dim, 160, 160]
        level3_out3 = self.lpm_320(level3_feat3)  # [B, projection_dim, 160, 160]
        level3_out4 = self.lpm_320(level3_feat4)  # [B, projection_dim, 160, 160]
        
        # Concat level3 features only and apply 1x1 conv
        level3_concat = torch.cat([level3_out1, level3_out2, level3_out3, level3_out4], dim=1)  # [B, projection_dim*4, 160, 160]
        level3_processed = self.conv_level3(level3_concat)  # [B, projection_dim, 160, 160]
        
        # Upsample level2 result to 160x160
        level2_up = F.interpolate(level2_refined, size=(160, 160), mode='bilinear', align_corners=False)  # [B, projection_dim, 160, 160]
        
        # Element-wise addition with upsampled level2
        level3_fused = level3_processed + level2_up  # [B, projection_dim, 160, 160]
        
        # Apply final FRM
        final_output = self.frm_level3(level3_fused, training)  # [B, projection_dim, 160, 160]
        
        # Final convolution
        final_output = self.final_conv(final_output)  # [B, projection_dim, 160, 160]
        
        return final_output
    
    
class Wavelet_v1(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(Wavelet_v1, self).__init__()
        # Define depthwise separable convolution layers for each band
        # LL band
        self.dw_conv_LL_1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, padding=1, groups=in_nc)
        self.pw_conv_LL_1 = nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=1)
        self.dw_conv_LL_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=3, padding=1, groups=out_nc)
        self.pw_conv_LL_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=1)
        
        # LH band
        self.dw_conv_LH_1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, padding=1, groups=in_nc)
        self.pw_conv_LH_1 = nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=1)
        self.dw_conv_LH_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=3, padding=1, groups=out_nc)
        self.pw_conv_LH_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=1)
        
        # HL band
        self.dw_conv_HL_1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, padding=1, groups=in_nc)
        self.pw_conv_HL_1 = nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=1)
        self.dw_conv_HL_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=3, padding=1, groups=out_nc)
        self.pw_conv_HL_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=1)
        
        # HH band
        self.dw_conv_HH_1 = nn.Conv2d(in_channels=in_nc, out_channels=in_nc, kernel_size=3, padding=1, groups=in_nc)
        self.pw_conv_HH_1 = nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=1)
        self.dw_conv_HH_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=3, padding=1, groups=out_nc)
        self.pw_conv_HH_2 = nn.Conv2d(in_channels=out_nc, out_channels=out_nc, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.DWT = DWTForward(J=1, wave='haar', mode='zero')
        self.IDWT = DWTInverse(wave='haar', mode='zero')
        
    def forward(self, x, training):
        
        if training:
            self.DWT = self.DWT.to(torch.float16)
            self.IDWT = self.IDWT.to(torch.float16)
        else: 
            self.DWT = self.DWT.to(torch.float32)
            self.IDWT = self.IDWT.to(torch.float32)

        Yl, Yh = self.DWT(x)
    
        # Process LL band with depthwise separable conv blocks
        LL = self.dw_conv_LL_1(Yl)
        LL = self.pw_conv_LL_1(LL)
        LL = self.relu(LL)
        LL = self.dw_conv_LL_2(LL)
        LL = self.pw_conv_LL_2(LL)
        LL_att = self.gap(LL)
        LL = LL * LL_att
        
        # Process LH band with depthwise separable conv blocks
        LH = self.dw_conv_LH_1(Yh[0][:, :, 0, :, :])
        LH = self.pw_conv_LH_1(LH)
        LH = self.relu(LH)
        LH = self.dw_conv_LH_2(LH)
        LH = self.pw_conv_LH_2(LH)
        LH_att = self.gap(LH)
        LH = LH * LH_att
        
        # Process HL band with depthwise separable conv blocks
        HL = self.dw_conv_HL_1(Yh[0][:, :, 1, :, :])
        HL = self.pw_conv_HL_1(HL)
        HL = self.relu(HL)
        HL = self.dw_conv_HL_2(HL)
        HL = self.pw_conv_HL_2(HL)
        HL_att = self.gap(HL)
        HL = HL * HL_att
        
        # Process HH band with depthwise separable conv blocks
        HH = self.dw_conv_HH_1(Yh[0][:, :, 2, :, :])
        HH = self.pw_conv_HH_1(HH)
        HH = self.relu(HH)
        HH = self.dw_conv_HH_2(HH)
        HH = self.pw_conv_HH_2(HH)
        HH_att = self.gap(HH)
        HH = HH * HH_att
        
        # Prepare for IDWT
        LH = LH.unsqueeze(2)
        HL = HL.unsqueeze(2)
        HH = HH.unsqueeze(2)
        Low = LL
        High = torch.cat([LH, HL, HH], dim=2)
        
        # Apply IDWT
        out = self.IDWT((Low, [High]))
            
        out = out + x  # skip connection
        return out


class AggregationNetwork(nn.Module):
    """
    Module for aggregating feature maps across time and space.
    Design inspired by the Feature Extractor from ODISE (Xu et. al., CVPR 2023).
    https://github.com/NVlabs/ODISE/blob/5836c0adfcd8d7fd1f8016ff5604d4a31dd3b145/odise/modeling/backbone/feature_extractor.py
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device

        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            self.mixing_weights_names.append(f"layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = None
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            if output_feature is None:
                output_feature = bottlenecked_feature
            else:
                output_feature += bottlenecked_feature
        return output_feature
    
# Original AggregationNetwork2
class AggregationNetwork2(nn.Module):
    """
    Similar to AggregationNetwork(above). Rather, it concatenate feature maps
    """
    def __init__(
            self, 
            feature_dims, 
            device, 
            projection_dim=384, 
            num_norm_groups=32,
            num_res_blocks=1, 
        ):
        super().__init__()
        self.bottleneck_layers = nn.ModuleList()
        self.feature_dims = feature_dims    
        # For CLIP symmetric cross entropy loss during training
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        self.device = device
        self.final_1x1_conv = nn.Conv2d(in_channels=projection_dim * len(feature_dims), out_channels=projection_dim, kernel_size=1, stride=1)
        
        self.mixing_weights_names = []
        for l, feature_dim in enumerate(self.feature_dims):
            bottleneck_layer = nn.Sequential(
                *ResNet.make_stage(
                    BottleneckBlock,
                    num_blocks=num_res_blocks,
                    in_channels=feature_dim,
                    bottleneck_channels=projection_dim // 4,
                    out_channels=projection_dim,
                    norm="GN",
                    num_norm_groups=num_norm_groups
                )
            )
            self.bottleneck_layers.append(bottleneck_layer)
            self.mixing_weights_names.append(f"layer-{l+1}")
        
        self.bottleneck_layers = self.bottleneck_layers.to(device)
        mixing_weights = torch.ones(len(self.bottleneck_layers))
        self.mixing_weights = nn.Parameter(mixing_weights.to(device))

    def forward(self, batch):
        """
        Assumes batch is shape (B, C, H, W) where C is the concatentation of all layer features.
        """
        output_feature = []
        start = 0
        mixing_weights = torch.nn.functional.softmax(self.mixing_weights)
        for i in range(len(mixing_weights)):
            # Share bottleneck layers across timesteps
            bottleneck_layer = self.bottleneck_layers[i % len(self.feature_dims)]
            # Chunk the batch according the layer
            # Account for looping if there are multiple timesteps
            end = start + self.feature_dims[i % len(self.feature_dims)]
            feats = batch[:, start:end, :, :]
            start = end
            # Downsample the number of channels and weight the layer
            bottlenecked_feature = bottleneck_layer(feats)
            bottlenecked_feature = mixing_weights[i] * bottlenecked_feature
            output_feature.append(bottlenecked_feature)
        
        if len(output_feature) != 1:
            output_feature = torch.cat(output_feature, dim=1)
            output_feature = self.final_1x1_conv(output_feature)
        else:
            output_feature = self.final_1x1_conv(output_feature[0])
        
        return output_feature

class SimpleAggregationNetwork_v2(nn.Module):
    def __init__(self, input_channels=3840, output_channels=512, hidden_dim=1024):
        super(SimpleAggregationNetwork_v2, self).__init__()
        
        # Grouped convolution to reduce input channels in a lightweight manner
        self.grouped_conv = nn.Conv2d(input_channels, hidden_dim, kernel_size=1, groups=8, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Pointwise convolution to bring down to 512 output channels
        self.pointwise_conv = nn.Conv2d(hidden_dim, output_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
    def forward(self, x):
        # x shape: (B, 3840, 160, 160)
        x = self.grouped_conv(x)  # Reduces channels from 3840 to hidden_dim
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise_conv(x)  # Reduces channels from hidden_dim to output_channels
        x = self.bn2(x)
        
        return x  # Output shape: (B, 512, 160, 160)



class DetectionAggregationNetwork(nn.Module):
    def __init__(self, projection_dim, feature_dims, device):
        super(DetectionAggregationNetwork, self).__init__()
        self.proj = nn.Conv2d(feature_dims[0], projection_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(projection_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dw_conv = nn.Conv2d(
            projection_dim, 
            projection_dim, 
            kernel_size=3, 
            padding=1, 
            groups=projection_dim,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(projection_dim)
        self.to(device)
        
    def forward(self, x):
        x = self.proj(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dw_conv(x)
        x = self.bn2(x)
        return x



"""
Functions for building the BottleneckBlock from Detectron2.
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/resnet.py
"""

def get_norm(norm, out_channels, num_norm_groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(num_norm_groups, channels),
        }[norm]
    return norm(out_channels)

class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.
    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
    
class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="GN",
        stride_in_1x1=False,
        dilation=1,
        num_norm_groups=32
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels, num_norm_groups),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels, num_norm_groups),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels, num_norm_groups),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out
    
class ResNet(nn.Module):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, stem, stages, num_classes=None, out_features=None, freeze_at=0):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
            freeze_at (int): The number of stages at the beginning to freeze.
                see :meth:`freeze` for detailed explanation.
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            # Avoid keeping unused layers in this module. They consume extra memory
            # and may cause allreduce to fail
            num_stages = max(
                [{"res2": 1, "res3": 2, "res4": 3, "res5": 4}.get(f, 0) for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            self._out_feature_strides[name] = current_stride = int(
                current_stride * np.prod([k.stride for k in blocks])
            )
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels
        self.stage_names = tuple(self.stage_names)  # Make it static for scripting

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))
        self.freeze(freeze_at)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.
        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of
                `block_class`. If the argument name is "xx_per_block", the
                argument is a list of values to be passed to each block in the
                stage. Otherwise, the same argument is passed to every block
                in the stage.
        Returns:
            list[CNNBlockBase]: a list of block module.
        Examples:
        ::
            stage = ResNet.make_stage(
                BottleneckBlock, 3, in_channels=16, out_channels=64,
                bottleneck_channels=16, num_groups=1,
                stride_per_block=[2, 1, 1],
                dilations_per_block=[1, 1, 2]
            )
        Usually, layers that produce the same feature map spatial size are defined as one
        "stage" (in :paper:`FPN`). Under such definition, ``stride_per_block[1:]`` should
        all be 1.
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith("_per_block"):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"same length as num_blocks={num_blocks}."
                    )
                    newk = k[: -len("_per_block")]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}!"
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v

            blocks.append(
                block_class(in_channels=in_channels, out_channels=out_channels, **curr_kwargs)
            )
            in_channels = out_channels
        return blocks

    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth (one of 18, 34, 50, 101, 152).
        If it doesn't create the ResNet variant you need, please use :meth:`make_stage`
        instead for fine-grained customization.
        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept
                `bottleneck_channels` argument for depth > 50.
                By default it is BasicBlock or BottleneckBlock, based on the
                depth.
            kwargs:
                other arguments to pass to `make_stage`. Should not contain
                stride and channels, as they are predefined for each depth.
        Returns:
            list[list[CNNBlockBase]]: modules in all stages; see arguments of
                :class:`ResNet.__init__`.
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else:
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for (n, s, i, o) in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs["bottleneck_channels"] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret