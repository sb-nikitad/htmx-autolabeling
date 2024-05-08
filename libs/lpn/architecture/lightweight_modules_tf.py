import tensorflow as tf
from tensorflow.keras import layers, Sequential
import tensorflow.keras as keras

BN_MOMENTUM = 0.1


class GCBlock(layers.Layer):

    def __init__(self, inplanes, planes, pool='att', fusions='channel_add'):
        super(GCBlock, self).__init__()
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions

        self.conv_mask = layers.Conv2D(1, kernel_size=1,
                                       kernel_initializer=keras.initializers.HeNormal(),
                                       bias_initializer=keras.initializers.Zeros())
        self.softmax = layers.Softmax(axis=2)

        self.channel_add_conv = Sequential([
            layers.Conv2D(self.planes, kernel_size=1,
                          kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001),
                          bias_initializer=keras.initializers.Zeros()),
            layers.LayerNormalization(axis=[1, 2, 3]),
            layers.Activation('relu'),
            layers.Conv2D(self.inplanes, kernel_size=1,
                          kernel_initializer=keras.initializers.Zeros(),
                          bias_initializer=keras.initializers.Zeros())]
        )

    def spatial_pool(self, x):
        # # [N, C, H, W]
        # batch, channel, height, width = x.shape
        #
        # input_x = x
        # # [N, C, H * W]
        # input_x = tf.reshape(input_x, [batch, channel, height * width])
        # # [N, 1, C, H * W]
        # input_x = tf.expand_dims(input_x, 1)
        # # [N, 1, H, W]
        # context_mask = self.conv_mask(x)
        # # [N, 1, H * W]
        # context_mask = tf.reshape(context_mask, [batch, 1, height * width])
        # # [N, 1, H * W]
        # context_mask = self.softmax(context_mask)
        # # [N, 1, H * W, 1]
        # context_mask = tf.expand_dims(context_mask, 3)
        # # [N, 1, C, 1]
        # context = tf.linalg.matmul(input_x, context_mask)
        # # [N, C, 1, 1]
        # context = tf.reshape(context, [batch, channel, 1, 1])

        # [N, H, W, C]
        batch, height, width, channel = x.shape

        input_x = x
        # [N, H * W, C]
        input_x = tf.reshape(input_x, [batch, height * width, channel])
        # print("1: ", input_x.shape)
        # [N, H * W, C, 1]
        input_x = tf.expand_dims(input_x, -1)
        # print("2: ", input_x.shape)
        # [N, H, W, 1]
        context_mask = self.conv_mask(x)
        # print("3: ", context_mask.shape)
        # [N, H * W, 1]
        context_mask = tf.reshape(context_mask, [batch, height * width, 1])
        # print("4: ", context_mask.shape)
        # [N, H * W, 1]
        context_mask = self.softmax(context_mask)
        # print("5: ", context_mask.shape)
        # [N, H * W, 1, 1]
        context_mask = tf.expand_dims(context_mask, -1)
        # print("6: ", context_mask.shape)
        # [N, C, 1, 1]
        context = keras.layers.dot([input_x, context_mask], axes=1)
        # print("7: ", context.shape)
        # [N, C, 1, 1]
        context = tf.reshape(context, [batch, 1, 1, channel])
        # print("8: ", context.shape)
        return context

    def call(self, x, training=None, **kwargs):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


class LW_Bottleneck(layers.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='GC'):
        super(LW_Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=1, use_bias=False,
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001))
        self.bn1 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                             gamma_initializer=keras.initializers.Ones(),
                                             beta_initializer=keras.initializers.Zeros())
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False,
                                            depthwise_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001))
        self.bn2 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                             gamma_initializer=keras.initializers.Ones(),
                                             beta_initializer=keras.initializers.Zeros())
        self.conv3 = layers.Conv2D(planes * self.expansion, kernel_size=1, use_bias=False,
                                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.001))
        self.bn3 = layers.BatchNormalization(momentum=BN_MOMENTUM,
                                             gamma_initializer=keras.initializers.Ones(),
                                             beta_initializer=keras.initializers.Zeros())
        self.relu = layers.Activation('relu')

        self.downsample = downsample
        self.stride = stride

        out_planes = planes * self.expansion // 16 if planes * self.expansion // 16 >= 16 else 16
        
        ####### Test remove attention modules #######
        #self.att = GCBlock(planes * self.expansion, out_planes, 'att', 'channel_add')
        self.att = None
        #############################################

    def call(self, x, training=None, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
