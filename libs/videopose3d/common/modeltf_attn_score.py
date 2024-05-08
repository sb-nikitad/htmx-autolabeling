# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential


class TemporalModelBaseTF(keras.Model):
    """
    Do not instantiate this class.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels, oks_branch=True):
        super().__init__()

        # Validate input
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = layers.Dropout(dropout)
        self.relu = layers.Activation('relu')

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = layers.BatchNormalization(momentum=0.1)
        self.shrink = layers.Conv1D(filters=num_joints_out * 3, kernel_size=1)
        self.oks_branch = oks_branch

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def call(self, x):
        assert len(x.shape) == 4
        # assert x.shape[-2] == self.num_joints_in
        # assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = tf.reshape(x, [x.shape[0], x.shape[1], -1])

        if self.oks_branch:
            skeleton, scores = self._forward_blocks(x)

            skeleton = tf.reshape(skeleton, [sz[0], -1, self.num_joints_out, 3])
            scores = tf.reshape(scores, [sz[0], -1, self.num_joints_out])

            return skeleton, scores
        else:
            skeleton = self._forward_blocks(x)
            skeleton = tf.reshape(skeleton, [sz[0], -1, self.num_joints_out, 3])
            return skeleton


class TemporalModelTF(TemporalModelBaseTF):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, oks_branch=True, dense=False,
                 model_type='all_sigmoid'):
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels,
                         oks_branch)

        self.model_type = model_type

        if 'sigmoid' in self.model_type:
            self.expand_conv_weight = layers.Conv1D(filters=channels, kernel_size=filter_widths[0], use_bias=False)
            self.expand_conv_value = layers.Conv1D(filters=channels, kernel_size=filter_widths[0], use_bias=False)
            self.expand_bn_weight = layers.BatchNormalization(momentum=0.1)
            self.expand_bn_value = layers.BatchNormalization(momentum=0.1)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        if len(filter_widths) == 5:
            self.slice_size = [241, 235, 217, 163]
        elif len(filter_widths) == 4:
            self.slice_size = [79, 73, 55]
        else:
            self.slice_size = [25, 19]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(layers.Conv1D(filters=channels,
                                             kernel_size=filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                             dilation_rate=next_dilation if not dense else 1,
                                             use_bias=False))
            layers_bn.append(layers.BatchNormalization(momentum=0.1))
            layers_conv.append(layers.Conv1D(filters=channels, kernel_size=1, dilation_rate=1, use_bias=False))
            layers_bn.append(layers.BatchNormalization(momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = layers_conv
        self.layers_bn = layers_bn
        if self.oks_branch:
            self.score_branch = layers.Conv1D(filters=num_joints_out, kernel_size=1, activation='sigmoid')
        self.num_joints = num_joints_in

    def _forward_blocks(self, x):

        if self.model_type == 'all_sigmoid':
            weight = tf.keras.activations.sigmoid(self.expand_bn_weight(self.expand_conv_weight(x)))
            value = self.relu(self.expand_bn_value(self.expand_conv_value(x)))
            x = tf.keras.layers.Multiply()([weight, value])
        elif self.model_type == 'separate_sigmoid':
            coord_idx = []
            conf_idx = []
            for i in range(self.num_joints):
                coord_idx.extend([i * 3, i * 3 + 1])
                conf_idx.append(i * 3 + 2)

            x_coord = tf.gather(x, coord_idx, axis=2)  # x[:, coord_idx, : ]
            x_conf = tf.gather(x, conf_idx, axis=2)  # x[:, conf_idx, : ]

            weight = tf.keras.activations.sigmoid(self.expand_bn_weight(self.expand_conv_weight(x_conf)))
            value = self.relu(self.expand_bn_value(self.expand_conv_value(x_coord)))
            x = tf.keras.layers.Multiply()([weight, value])

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, pad + shift: self.slice_size[i] - pad + shift, :]

            num_layers = len(self.layers_conv) // (len(self.pad) - 1)
            for j in range(num_layers - 1):
                x = self.drop(self.relu(self.layers_bn[num_layers * i + j](self.layers_conv[num_layers * i + j](x))))

            x = res + self.drop(
                self.relu(self.layers_bn[num_layers * (i + 1) - 1](self.layers_conv[num_layers * (i + 1) - 1](x))))

        if self.oks_branch:
            skeleton = self.shrink(x)
            scores = self.score_branch(x)
            return skeleton, scores
        else:
            return self.shrink(x)
