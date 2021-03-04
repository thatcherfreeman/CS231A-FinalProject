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
        self.encoder = E_resnet(pretrained=True, resnet_type='resnet18')
        block_channels = [64, 128, 256, 512]
        self.decoder = SkipDecoder(block_channels)

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

class Hu18(nn.Module):
    '''
    Adapted model from hu et al, with resnet 18 encoder.
    '''
    def __init__(self):
        super(Hu18, self).__init__()
        self.encoder = E_resnet(pretrained=True, resnet_type='resnet18')
        block_channels = [64, 128, 256, 512]
        self.decoder = Decoder(block_channels)
        self.mff = MFF(block_channels)
        self.refinement = Refinement(block_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block1, block2, block3, block4 = self.encoder(x)
        decoder_output = self.decoder(block4)
        mff_output = self.mff(block1, block2, block3, block4)
        out = self.refinement(decoder_output, mff_output)
        return out


class Hu34(nn.Module):
    '''
    Adapted model from hu et al, with resnet 34 encoder.
    '''
    def __init__(self):
        super(Hu34, self).__init__()
        self.encoder = E_resnet(pretrained=True, resnet_type='resnet34')
        block_channels = [64, 128, 256, 512]
        self.decoder = Decoder(block_channels)
        self.mff = MFF(block_channels)
        self.refinement = Refinement(block_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block1, block2, block3, block4 = self.encoder(x)
        decoder_output = self.decoder(block4)
        mff_output = self.mff(block1, block2, block3, block4)
        out = self.refinement(decoder_output, mff_output)
        return out


class Hu50(nn.Module):
    '''
    Adapted model from hu et al, with resnet 50 encoder.
    '''
    def __init__(self):
        super(Hu50, self).__init__()
        self.encoder = E_resnet(pretrained=True, resnet_type='resnet50')
        block_channels = [256, 512, 1024, 2048]
        self.decoder = Decoder(block_channels)
        self.mff = MFF(block_channels)
        self.refinement = Refinement(block_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block1, block2, block3, block4 = self.encoder(x)
        decoder_output = self.decoder(block4)
        mff_output = self.mff(block1, block2, block3, block4)
        out = self.refinement(decoder_output, mff_output)
        return out


# Adapted from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L40-L66
class E_resnet(nn.Module):
    '''
    Standard Resnet18 implementation, excluding the output layers.
    Modified to output intermediate layer results.
    '''
    def __init__(self, pretrained: bool = True, resnet_type: str = 'resnet18'):
        super(E_resnet, self).__init__()
        if resnet_type == 'resnet18':
            original_model = resnet.resnet18(pretrained=pretrained)
        elif resnet_type == 'resnet50':
            original_model = resnet.resnet50(pretrained=pretrained)
        elif resnet_type == 'resnet34':
            original_model = resnet.resnet34(pretrained=pretrained)

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
    def __init__(self, block_channels: Tuple[int, int, int, int]):
        super(SkipDecoder, self).__init__()
        self.conv = nn.Conv2d(block_channels[3], block_channels[2], kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(block_channels[2])
        self.up1 = UpProjection(block_channels[2] * 2, block_channels[1], 2)
        self.up2 = UpProjection(block_channels[1] * 2, block_channels[0], 2)
        self.up3 = UpProjection(block_channels[0] * 2, block_channels[0], 2)
        self.up4 = UpProjection(block_channels[0], block_channels[0], 2)
        self.conv2 = nn.Conv2d(block_channels[0], block_channels[0], kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_channels[0])
        self.conv3 = nn.Conv2d(block_channels[0], 1, kernel_size=3, padding=1, stride=1)

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
        x1 = self.up1(x0, block3) # N, 128, 60, 80
        x2 = self.up2(x1, block2) # N, 64, 120, 160
        x3 = self.up3(x2, block1) # N, 64, 240, 320
        x4 = self.up4(x3) # N, 64, 480, 640
        out = F.relu(self.bn2(self.conv2(x4))) # N, 64, 480, 640
        out = self.conv3(out) # N, 1, 480, 640
        return out

# Based on the decoder class here
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L118-L151
class Decoder(nn.Module):
    '''
    Decoder without skip connections from encoder blocks
    '''
    def __init__(self, block_channels: Tuple[int, int, int, int]):
        super(Decoder, self).__init__()
        self.up1 = UpProjection(block_channels[3], block_channels[2], scale=2)
        self.up2 = UpProjection(block_channels[2], block_channels[1], scale=2)
        self.up3 = UpProjection(block_channels[1], block_channels[0], scale=2)
        self.up4 = UpProjection(block_channels[0], block_channels[0], scale=4, ksize=5)

    def forward(
        self,
        block4: torch.Tensor
    ) -> torch.Tensor:
        '''
        Inputs:
            block4 shape (N, 512, 15, 20)
        Outputs:
            Depth map of shape (N, 1, 480, 640)
        '''
        # x0 = F.relu(self.bn(self.conv(block4))) # N, 512, 15, 20
        # x0 = F.interpolate(x0, scale_factor=2, align_corners=True, mode='bilinear') # N, 512, 30, 40

        # x0 = F.interpolate(block4, scale_factor=2, align_corners=True, mode='bilinear') # N, 512, 30, 40
        x1 = self.up1(block4) # N, 256, 30, 40
        x2 = self.up2(x1) # N, 128, 60, 80
        x3 = self.up3(x2) # N, 64, 120, 160
        x4 = self.up4(x3) # N, 64, 480, 640
        return x4


# Copied from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L13-L38
class UpProjection(nn.Sequential):
    def __init__(self, input_features: int, output_features: int, scale: int=2, ksize=3):
        super(UpProjection, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_features, output_features,
                               kernel_size=ksize, stride=1, padding=ksize//2, bias=False)
        self.bn1 = nn.BatchNorm2d(output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(output_features, output_features,
                                 kernel_size=ksize, stride=1, padding=ksize//2, bias=False)
        self.bn1_2 = nn.BatchNorm2d(output_features)
        self.conv2 = nn.Conv2d(input_features, output_features,
                               kernel_size=ksize, stride=1, padding=ksize//2, bias=False)
        self.bn2 = nn.BatchNorm2d(output_features)

    def forward(self, x: torch.Tensor, block: Optional[torch.Tensor]=None) -> torch.Tensor:
        '''
        Inputs:
            x shape N, C, H, W
            block shape N, C, H, W
        Outputs:
            out shape N, C', H*scale, W*scale
        '''
        if block is not None:
            x = torch.cat([x, block], dim=1) # concatenate along channels

        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

# Copied from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L153-L185
class MFF(nn.Module):

    def __init__(self, block_channel: Tuple[int,int,int,int], num_features=64):

        super(MFF, self).__init__()

        self.up1 = UpProjection(
            input_features=block_channel[0], output_features=num_features // 4, scale=4, ksize=3)

        self.up2 = UpProjection(
            input_features=block_channel[1], output_features=num_features // 4, scale=8, ksize=3)

        self.up3 = UpProjection(
            input_features=block_channel[2], output_features=num_features // 4, scale=16, ksize=3)

        self.up4 = UpProjection(
            input_features=block_channel[3], output_features=num_features // 4, scale=32, ksize=3)

        self.conv1 = nn.Conv2d(
            num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)


    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_m1 = self.up1(x_block1)
        x_m2 = self.up2(x_block2)
        x_m3 = self.up3(x_block3)
        x_m4 = self.up4(x_block4)

        x = self.bn(self.conv1(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x

# Copied from
# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/models/modules.py#L188-L216
class Refinement(nn.Module):
    def __init__(self, block_channel:Tuple[int,int,int,int]):

        super(Refinement, self).__init__()

        num_features = 64 + block_channel[0]
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=1, bias=True)

    def forward(self, decoder_output, mff_output):
        x = torch.cat([decoder_output, mff_output], dim=1)
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)

        return x2


def get_model(model_name: str) -> type:
    models = [Baseline, Hu18, Hu50, Hu34]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'
