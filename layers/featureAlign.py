import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D

class FeatureAlign(tf.keras.layers.Layer):
    def __init__(self,
                 num_filters,
                 kernel_size=3,
                 flag_norm=True):
        super(FeatureAlign, self).__init__()
        self.null_mask = tf.zeros((0, 0, 0, 0))
        channels_ = 2 * kernel_size * kernel_size
        self.conv_offset = tf.keras.layers.Conv2D(channels_,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.0),
            bias_initializer='zeros',
            name='conv_offset_mask')

        # grouping not supported in DCN as of now.
        self.conv_adaption = DeformableConv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            dilation_rate=1,
            weight_groups=1,
            offset_groups=1,
            use_mask=False,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)
        )
        self.relu = tf.nn.relu()
        self.norm =  tfa.layers.GroupNormalization(32)
        self.flag_norm = flag_norm

    def forward(self, x, y):
        offset = self.conv_offset(y)
        x = tf.transpose(x, (0, 3, 1, 2))
        offset = tf.transpose(offset, (0, 3, 1, 2))

        if self.flag_norm:
            x = self.relu(
                self.norm(self.conv_adaption([x, offset, self.null_mask])))
        else:
            x = self.relu(self.conv_adaption([x, offset, self.null_mask]))
        return tf.transpose(out, (0, 2, 3, 1))

