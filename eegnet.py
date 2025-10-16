from torch import nn
import torch
import torch.nn.functional as F

class SE_Block(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class EEGNet(nn.Module):
    def __init__(
        self, sampling_fq, data_shape, n_classes, dropout=0.50, F1=8, D=2, F2=16, pool_size=(4, 8), layernorm=False, sae=False
    ):
        super(EEGNet, self).__init__()

        self.chans = data_shape[0]
        self.samples = data_shape[1]
        self.kernLength = sampling_fq // 2
        
        if sae:
            self.se_block = SE_Block(F2, 8)

        # BLOCK 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, self.kernLength),
            padding="same",
            bias=False,
        )
        if layernorm:
            self.norm1 = nn.LayerNorm([self.samples])
        else:
            self.norm1 = nn.BatchNorm2d(F1, affine=True)
        self.depthwiseconv1 = nn.Conv2d(
            in_channels=F1,
            out_channels=D * F1,
            kernel_size=(self.chans, 1),
            groups=F1,
            padding="valid",
            bias=False,
        )
        if layernorm:
            self.norm2 = nn.LayerNorm([self.samples])
        else:
            self.norm2 = nn.BatchNorm2d(D * F1, affine=True)
        self.pooling1 = nn.AvgPool2d((1, pool_size[0]))

        # BLOCK 2
        self.depthwiseconv2 = nn.Conv2d(
            in_channels=D * F1,
            out_channels=D * D * F1,
            kernel_size=(1, 16),
            groups=D * F1,
            padding="same",
            bias=False,
        )
        self.pointwiseconv1 = nn.Conv2d(
            in_channels=D * D * F1, out_channels=F2, kernel_size=1, padding="same", bias=False
        )
        if layernorm:
            self.norm3 = nn.LayerNorm([self.samples//4])
        else:
            self.norm3 = nn.BatchNorm2d(F2, affine=True)
  
        self.pooling2 = nn.AvgPool2d((1, pool_size[1]))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=F2 * (self.samples // (pool_size[0] * pool_size[1])), out_features=n_classes
        )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        # BLOCK 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.depthwiseconv1(x)
        x = self.norm2(x)

        x = self.activation(x)
        x = self.pooling1(x)
        x = self.dropout(x)

        # BLOCK 2
        x = self.depthwiseconv2(x)
        x = self.pointwiseconv1(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.pooling2(x)
        if hasattr(self, 'se_block'):
            x = self.se_block(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.linear(x)

        return x

