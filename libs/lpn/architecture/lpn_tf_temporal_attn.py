import os
import logging
import math
# import torch
# import torch.nn as nn
from .lightweight_modules_tf import LW_Bottleneck
from .temporal_attn import Temporal_Attn

import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class Golf_LPN_TA(keras.Model):

    def __init__(self, block, layer_dims, in_frame=3, concat=False, conv2d_attn=False, depthwise=False, **kwargs):
        super(Golf_LPN_TA, self).__init__()
        # extra = cfg.MODEL.EXTRA

        self.inplanes = 64
        self.deconv_with_bias = False
        self.attention = 'GC'
        self.in_frame = in_frame
        self.conv2d_attn = conv2d_attn
        self.concat = concat
        self.depthwise = depthwise

        self.ta_module = Temporal_Attn(in_channel=64, in_frame=in_frame, concat=self.concat, conv2d_attn=conv2d_attn, depth_wise=self.depthwise)
        self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', use_bias=False,
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001))
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                             gamma_initializer=keras.initializers.Ones(),
                                             beta_initializer=keras.initializers.Zeros())
        self.relu = layers.Activation('relu')
        self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.layer1 = self._make_layer(block, 64, layer_dims[0])
        self.layer2 = self._make_layer(block, 128, layer_dims[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layer_dims[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layer_dims[3], stride=1)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            2,
            [256, 256],
            [4, 4],
        )

        ########## use 2 heads for human body kpts and golf club kpts seperately ###########
        self.human_kpts_head = layers.Conv2D(
            34,
            kernel_size=1,
            strides=1,
            padding = 'same',
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001)
        )

        self.golf_club_kpts_head = layers.Conv2D(
            4,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001)
        )
        ####################################################################################

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential([
                layers.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False,
                              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001)),
                layers.BatchNormalization(momentum=BN_MOMENTUM,
                                          gamma_initializer=keras.initializers.Ones(),
                                          beta_initializer=keras.initializers.Zeros())
            ])

        block_layers = Sequential()
        block_layers.add(block(self.inplanes, planes, stride, downsample, self.attention))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_layers.add(block(self.inplanes, planes, attention=self.attention))

        return block_layers

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

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        block_layers = Sequential()
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            block_layers.add(Sequential([
                layers.UpSampling2D(size=2),
                layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False,
                                       depthwise_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001)),
                layers.BatchNormalization(momentum=BN_MOMENTUM,
                                          gamma_initializer=keras.initializers.Ones(),
                                          beta_initializer=keras.initializers.Zeros()),
                layers.Activation('relu'),
                layers.Conv2D(filters=planes, kernel_size=1, padding='same', use_bias=False,
                              kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001)),
                layers.BatchNormalization(momentum=BN_MOMENTUM,
                                          gamma_initializer=keras.initializers.Ones(),
                                          beta_initializer=keras.initializers.Zeros()),
                layers.Activation('relu')
            ]))

            self.inplanes = planes

        return block_layers

    def call(self, x, training=None, **kwargs):
        lst = []
        for i in range(self.in_frame):
            tmp = self.conv1(x[:, :, :, :, i])
            tmp = self.bn1(tmp)
            tmp = self.relu(tmp)
            tmp = self.maxpool(tmp)
            lst.append(tmp)

        x = self.ta_module(lst)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.deconv_layers(x)
        #        x = self.final_layer(features)
        ########## use 2 heads for human body kpts and golf club kpts seperately ###########
        x1 = self.human_kpts_head(features)
        x2 = self.golf_club_kpts_head(features)
        x = tf.concat([x1, x2], axis=3)
        ####################################################################################

        return x


resnet_spec = {
    50: (LW_Bottleneck, [3, 4, 6, 3]),
    101: (LW_Bottleneck, [3, 4, 23, 3]),
    152: (LW_Bottleneck, [3, 8, 36, 3])
}


def get_pose_net():
    num_layers = 50
    block_class, layers = resnet_spec[num_layers]

    model = Golf_LPN_TA(block_class, layers, conv2d_attn=True, depthwise=True)
    model.build(input_shape=(1, 256, 192, 3, 3))

    return model
