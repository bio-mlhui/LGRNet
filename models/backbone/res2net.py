import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from detectron2.modeling import BACKBONE_REGISTRY
from einops import rearrange, reduce, repeat
from .utils import VideoMultiscale_Shape

class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width      = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width*scale)
        self.nums  = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs, bns = [], []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp  = spx[i] if i == 0 or self.stype == 'stage' else sp + spx[i]
            sp  = self.convs[i](sp)
            sp  = F.relu(self.bns[i](sp), inplace=True)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class Res2Net(nn.Module):
    def __init__(self, layers, snapshot, baseWidth=26, scale=4):
        super(Res2Net, self).__init__()
        self.inplanes  = 64
        self.snapshot  = snapshot
        self.baseWidth = baseWidth
        self.scale     = scale
        self.conv1     = nn.Sequential(
                                nn.Conv2d(3, 32, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32, 64, 3, 1, 1, bias=False)
                            )
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, 512, layers[3], stride=2)

        state_dict:dict = torch.load(self.snapshot, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        del state_dict

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers        = [block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load(self.snapshot), strict=False)

@BACKBONE_REGISTRY.register()
class Res2Net_50_EachFrame(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        pt_path = os.getenv('PT_PATH')
        res2net = Res2Net([3, 4, 6, 3], os.path.join(pt_path, 'res2net/res2net50_v1b_26w_4s-3cf99910.pth'))
        self.res2net = res2net

        freeze = configs['freeze']

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

        self.multiscale_shapes = {}
        for name, temporal_stride, spatial_stride, dim  in zip(['res2', 'res3', 'res4', 'res5'],  
                                                               [1, 1, 1, 1], 
                                                               [4, 8, 16, 32],
                                                               [256, 512, 1024, 2048]):
            self.multiscale_shapes[name] =  VideoMultiscale_Shape(temporal_stride=temporal_stride, 
                                                                  spatial_stride=spatial_stride, dim=dim)

        self.max_stride = [1, 32]

    
    def forward(self, x):
        batch_size, _, T = x.shape[:3]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        layer_outputs = self.res2net(x)

        ret = {}
        names = ['res2', 'res3', 'res4', 'res5']
        for name, feat in zip(names, layer_outputs):
            ret[name] = rearrange(feat.contiguous(), '(b t) c h w -> b c t h w',b=batch_size, t=T)

        return ret
    

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)       
 