import tensorflow as tf
import tensorflow_addons as tfa
from layers.featureAlign import FeatureAlign

class FastMaskIoUNet(tf.keras.layers.Layer):
    def __init__(self, num_class):
        super(FastMaskIoUNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = tf.keras.layers.Conv2D(16, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv3 = tf.keras.layers.Conv2D(16, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv5 = tf.keras.layers.Conv2D(64, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv6 = tf.keras.layers.Conv2D(128, (3, 3), 2,
          kernel_initializer=tf.keras.initializers.HeNormal())
        self.mask_scoring = tf.keras.layers.Conv2D(num_class-1, (1, 1), 1,
          padding="valid",
          kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=0.001))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.mask_scoring(x)
        maskiou_p = tf.reduce_max(x, axis=(1,2))

        return maskiou_p

class PredictionModule(tf.keras.layers.Layer):

    def __init__(self, out_channels, num_class, stacked_convs=2):
        super(PredictionModule, self).__init__()

        self.norm_class_conv =  tfa.layers.GroupNormalization(32)
        self.norm_box_conv =  tfa.layers.GroupNormalization(32)
        self.class_conv = []
        self.box_conv = []

        for i in range(stacked_convs-1):
            self.class_conv.append(
              self.norm_class_conv(
                tf.keras.layers.Conv2D(out_channels, (3, 3), 1, 
                  padding="same",
                  kernel_initializer=tf.keras.initializers.VarianceScaling(
                    mode="fan_avg", distribution='uniform'),
                  use_bias=False)))

        for i in range(stacked_convs):
            self.box_conv.append(
              self.norm_box_conv(
                tf.keras.layers.Conv2D(out_channels, (3, 3), 1, 
                  padding="same",
                  kernel_initializer=tf.keras.initializers.VarianceScaling(
                    mode="fan_avg", distribution='uniform'),
                  use_bias=False)))


        self.fcos_cls = tf.keras.layers.Conv2D(num_class-1, (3, 3), 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))

        self.fcos_reg = tf.keras.layers.Conv2D(4, (3, 3), 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))

        self.fcos_centerness = tf.keras.layers.Conv2D(1, (3, 3), 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))
        self.feat_align = FeatureAlign(out_channels)

    def call(self, p):
        cls_feat = p
        reg_feat = p
        for cls_layer in self.class_conv:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.box_conv:
            reg_feat = reg_layer(reg_feat)

        # TODO: add trainable scale parameter here.
        bbox_pred = self.fcos_reg(reg_feat)
        cls_feat = self.feat_align(cls_feat, bbox_pred)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(reg_feat)
        cof_pred = self.sip_cof(cls_feat)

        return cof_pred, centerness, cls_score, bbox_pred, reg_feat
