from operator import mod
from xml.sax.xmlreader import AttributesNSImpl
from cv2 import sepFilter2D
from matplotlib.pyplot import axis
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, Sequential
import tensorflow as tf

BN_MOMENTUM = 0.1


class Temporal_Attn(keras.Model):
    def __init__(self, in_channel=64, in_frame=3, concat=False, conv2d_attn=False, depth_wise=False):
        super(Temporal_Attn, self).__init__()
        self.conv2d_attn = conv2d_attn
        self.in_frame = in_frame
        self.concat = concat
        self.in_channel = in_channel

        if self.conv2d_attn:
            if depth_wise:
                self.conv2d1 = layers.DepthwiseConv2D(3, padding='same', activation='relu')
            else:
                self.conv2d1 = layers.Conv2D(self.in_channel * 3, 3, padding='same', activation='relu')
            self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                                 gamma_initializer=keras.initializers.Ones(),
                                                 beta_initializer=keras.initializers.Zeros())
            self.conv2d2 = layers.Conv2D(self.in_channel, 3, padding='same', activation='relu')
            if self.concat:
                self.conv2d3 = layers.Conv2D(self.in_channel, 3, padding='same', activation='relu')
        else:
            self.conv3d1 = layers.Conv3D(self.in_frame, 3, padding='same', activation='relu')
            self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                                 gamma_initializer=keras.initializers.Ones(),
                                                 beta_initializer=keras.initializers.Zeros())
            self.conv3d2 = layers.Conv3D(self.in_frame, 3, padding='same', activation='relu')
            if self.concat:
                self.conv3d3 = layers.Conv3D(1, 3, padding='same', activation='relu')

    def call(self, x, training=None, **kwargs):
        if self.conv2d_attn:
            x = tf.concat(x, axis=-1)
            attn_map = self.conv2d1(x)
            attn_map = self.bn1(attn_map)
            attn_map = tf.nn.softmax(attn_map, axis=-1)
            attn_map = attn_map * x
            attn_map = self.conv2d2(attn_map)
            if self.concat:
                x = tf.concat([x[:, :, :, self.in_channel:2 * self.in_channel], attn_map], axis=-1)
                x = self.conv2d3(x)
            else:
                x = attn_map
        else:
            x = tf.stack(x, axis=-1)
            attn_map = self.conv3d1(x)
            attn_map = self.bn1(attn_map)
            attn_map = self.conv3d2(attn_map)
            attn_map = tf.nn.softmax(attn_map, axis=4)
            attn_map = attn_map * x
            attn_map = tf.reduce_sum(attn_map, axis=4)
            if self.concat:
                x = tf.stack([x[:, :, :, :, 1], attn_map], axis=-1)
                x = tf.squeeze(self.conv3d3(x), axis=-1)
            else:
                x = attn_map
        return x


if __name__ == '__main__':
    model = Temporal_Attn(conv2d_attn=True, concat=True, in_channel=4)
    x = np.zeros((1, 64, 48, 4))
    print(model([x, x, x]).shape)
