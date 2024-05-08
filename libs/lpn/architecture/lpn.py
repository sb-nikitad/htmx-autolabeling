import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lightweight_modules import LW_Bottleneck

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class LPN(nn.Module):

    def __init__(self, block, layers, **kwargs):
        super(LPN, self).__init__()
        # extra = cfg.MODEL.EXTRA

        self.inplanes = 64
        self.deconv_with_bias = False
        self.attention = "GC"

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=BN_MOMENTUM)

        self.inplanes = 256

        # used for deconv layers
        self.deconv_layers_0 = self._make_deconv_layer(
            0,
            [256, 256],
            [4, 4]
        )

        self.deconv_layers_1 = self._make_deconv_layer(
            1,
            [256, 256],
            [4, 4]
        )

#        self.final_layer = nn.Conv2d(
#            in_channels=extra.NUM_DECONV_FILTERS[-1],
#            out_channels=cfg.MODEL.NUM_JOINTS,
#            kernel_size=extra.FINAL_CONV_KERNEL,
#            stride=1,
#            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
#        )

        ########## use 2 heads for human body kpts and golf club kpts seperately ###########
        self.human_kpts_head = nn.Conv2d(
            in_channels=256,
            out_channels=33,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.golf_club_kpts_head = nn.Conv2d(
            in_channels=256,
            out_channels=6,
            kernel_size=1,
            stride=1,
            padding=0
        )
        ####################################################################################

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=self.attention))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding


    def _make_deconv_layer(self, i, num_filters, num_kernels):
        layers = []
        kernel, padding, output_padding = \
            self._get_deconv_cfg(num_kernels[i], i)

        planes = num_filters[i]
        layers.extend([
            # nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel,
            #                   stride=2, padding=padding, output_padding=output_padding, bias=self.deconv_with_bias),
            # nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(self.inplanes, planes, kernel_size=3, stride=1, padding=padding, groups=math.gcd(self.inplanes, planes), bias=self.deconv_with_bias),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ])
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = F.interpolate(x, size=(32, 24), mode='nearest')
        x = self.deconv_layers_0(x)
        x = F.interpolate(x, size=(64, 48), mode='nearest')
        features = self.deconv_layers_1(x)

#        x = self.final_layer(features)
        ########## use 2 heads for human body kpts and golf club kpts seperately ###########
        x1 = self.human_kpts_head(features)
        x2 = self.golf_club_kpts_head(features)
        x = torch.cat((x1, x2), dim = 1)
        ####################################################################################
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {
    50: (LW_Bottleneck, [3, 4, 6, 3]),
    101: (LW_Bottleneck, [3, 4, 23, 3]),
    152: (LW_Bottleneck, [3, 8, 36, 3])
}


def get_pose_net(**kwargs):
    num_layers = 50

    block_class, layers = resnet_spec[num_layers]

    model = LPN(block_class, layers, **kwargs)

    return model
