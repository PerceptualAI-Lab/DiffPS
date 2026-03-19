import torch.nn as nn
import pdb
import numpy as np
import torch

def bilinear_kernel(kernel_size, in_channels, out_channels):
    """
    Bilinear interpolation을 위한 커널을 생성.
    in_channels와 out_channels가 동일.
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    # 2D bilinear 필터 계산
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    
    # in_channels와 out_channels 각각에 대해 대각 행렬 형태로 배치
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    for i in range(in_channels):
        weight[i, i, :, :] = filt

    return torch.from_numpy(weight)


class LearnableUpsample(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 scale_factor=2,
                 kernel_size=4, 
                 stride=2, 
                 padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self._init_weights(kernel_size, in_channels, out_channels)
        
    def _init_weights(self, kernel_size, in_channels, out_channels):
        # bilinear 커널을 생성하여 weight를 초기화합니다.
        bilinear_weights = bilinear_kernel(kernel_size, in_channels, out_channels)
        self.conv_transpose.weight.data.copy_(bilinear_weights)
        if self.conv_transpose.bias is not None:
            nn.init.constant_(self.conv_transpose.bias, 0)
    
    
    def forward(self, x):
        return self.conv_transpose(x)
    

if __name__ == "__main__":
    
    upsample = LearnableUpsample(in_channels=640, out_channels=640)
    print(sum([p.numel() for p in upsample.parameters()]))
    pdb.set_trace()

