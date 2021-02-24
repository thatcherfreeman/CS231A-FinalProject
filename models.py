import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torchvision.models import resnet


class Baseline(nn.Module):
    '''
    Baseline U-net model, consisting of pretrained resnet encoder
    and a decoder with skip connections.
    '''
    def __init__(self):
        super(Baseline, self).__init__()
        self.encoder = E_resnet(pretrained=True)
        self.decoder = SkipDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Input:
            x shape N, C, H, W
        Output:
            y shape N, 1, H, W
        '''
        block1, block2, block3, block4 = self.encoder(x)
        out = self.decoder(block1, block2, block3, block4)
        return out


# Adapted from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L40-L66
class E_resnet(nn.Module):
    '''
    Standard Resnet50 implementation, excluding the output layers.
    Modified to output intermediate layer results.
    '''
    def __init__(self, pretrained: bool=False):
        super(E_resnet, self).__init__()
        original_model = resnet.resnet18(pretrained=pretrained)
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Input x of shape N, C, H, W
                        (N, 3, 480, 640)
        Outputs:
            block1 shape (N, 64, 120, 160)
            block2 shape (N, 128, 60, 80)
            block3 shape (N, 256, 30, 40)
            block4 shape (N, 512, 15, 20)
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return x_block1, x_block2, x_block3, x_block4


# Based on the decoder class here
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L118-L151
# However, altered to have skip connections and make a model output directly.
class SkipDecoder(nn.Module):
    '''
    Decoder with skip connections from encoder blocks
    '''
    def __init__(self):
        super(SkipDecoder, self).__init__()
        self.conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.up1 = UpProjection(256, 128)
        self.up2 = UpProjection(128, 64)
        self.up3 = UpProjection(64, 64)
        self.up4 = UpProjection(64, 64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2, stride=1)

    def forward(
        self,
        block1: torch.Tensor,
        block2: torch.Tensor,
        block3: torch.Tensor,
        block4: torch.Tensor
    ) -> torch.Tensor:
        '''
        Inputs:
            block1 shape (N, 64, 120, 160)
            block2 shape (N, 128, 60, 80)
            block3 shape (N, 256, 30, 40)
            block4 shape (N, 512, 15, 20)
        Outputs:
            Depth map of shape (N, 1, 480, 640)
        '''
        x0 = F.relu(self.bn(self.conv(block4))) # N, 256, 15, 20
        x0 = F.interpolate(x0, scale_factor=2, align_corners=True, mode='bilinear') # N, 256, 30, 40
        x1 = self.up1(x0 + block3, 2) # N, 128, 60, 80
        x2 = self.up2(x1 + block2, 2) # N, 64, 120, 160
        x3 = self.up3(x2 + block1, 2) # N, 64, 240, 320
        x4 = self.up4(x3, 2) # N, 64, 480, 640
        out = F.relu(self.bn2(self.conv2(x4))) # N, 64, 480, 640
        out = self.conv3(out) # N, 1, 480, 640
        return out


# Copied from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L13-L38
class UpProjection(nn.Sequential):
    def __init__(self, input_features: int, output_features: int):
        super(UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(input_features, output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(output_features, output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(output_features)
        self.conv2 = nn.Conv2d(input_features, output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(output_features)

    def forward(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        '''
        Inputs:
            x shape N, C, H, W
            scale int
        Outputs:
            out shape N, C', H*scale, W*scale
        '''
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out


def get_model(model_name: str) -> type:
    models = [Baseline]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'
