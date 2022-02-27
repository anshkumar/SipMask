import tensorflow as tf

from layers.fpn import FeaturePyramidNeck
from layers.head import PredictionModule, FastMaskIoUNet
import numpy as np
assert tf.__version__.startswith('2')
from detection import Detect
from data import anchor
from backbone import resnet

class SipMask(tf.keras.Model):
    """
        Creating the SipMask Architecture
        Arguments:

    """

    def __init__(self, img_h, img_w, fpn_channels, num_class, 
                strides=(4, 8, 16, 32, 64), use_dcn=False, 
                base_model_trainable=False, dcn_trainable=True):
        super(SipMask, self).__init__()
        self.strides = strides
        out = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']

        if not use_dcn:
            base_model = tf.keras.applications.resnet50.ResNet50(
                input_shape=(img_h,img_w,3),
                include_top=False,
                layers=tf.keras.layers,
                weights='imagenet')

            # Freeze the convolutional base
            base_model.trainable = base_model_trainable 
        else:
            base_model = resnet.ResNet50(
                input_shape=(img_h,img_w,3),
                include_top=False,
                layers=tf.keras.layers,
                weights='imagenet',
                dcn_layers=[False, True, True, True])

            # Freeze the convolutional base
            base_model.trainable = base_model_trainable 
            for layer in base_model.layers:
                if layer.name.startswith('dcn'):
                    layer.trainable =  dcn_trainable
        
        # extract certain feature maps for FPN
        self.backbone_resnet = tf.keras.Model(inputs=base_model.input,
                                              outputs=[
                                              base_model.get_layer(x).output 
                                              for x in out])
        
        # Calculating feature map size
        # https://stackoverflow.com/a/44242277/4582711
        # https://github.com/tensorflow/tensorflow/issues/4297#issuecomment-\
        # 246080982
        self.feature_map_size = np.array(
            [list(base_model.get_layer(x).output.shape[1:3]) for x in out])
        out_height_p6 = np.ceil(
            (self.feature_map_size[-1, 0]).astype(np.float32) / float(2))
        out_width_p6  = np.ceil(
            (self.feature_map_size[-1, 1]).astype(np.float32) / float(2))
        out_height_p7 = np.ceil(out_height_p6 / float(2))
        out_width_p7  = np.ceil(out_width_p6/ float(2))
        self.feature_map_size = np.concatenate(
            (self.feature_map_size, 
            [[out_height_p6, out_width_p6], [out_height_p7, out_width_p7]]), 
            axis=0)

        self.nc = 32
        self.backbone_fpn = FeaturePyramidNeck(fpn_channels)
        
        self.sip_cof = tf.keras.layers.Conv2D(self.nc*4, (3, 3), 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))

        self.sip_mask_lat = tf.keras.layers.Conv2D(self.nc, (3, 3), 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))
        self.sip_mask_lat0 = tf.keras.layers.Conv2D(fpn_channels*2, 1, 1, 
                padding="same",
                kernel_initializer=tf.keras.initializers.TruncatedNormal(
                  stddev=0.03))

        self.predictionHead = PredictionModule(256, num_class)
        self.fastMaskIoUNet = FastMaskIoUNet(num_class)

        # post-processing for evaluation
        self.detect = Detect(num_class, max_output_size=300, 
            per_class_max_output_size=100,
            conf_thresh=0.05, nms_thresh=0.5)
        self.max_output_size = 300

    @tf.function
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.keras.applications.resnet50.preprocess_input(inputs)

        # https://www.tensorflow.org/tutorials/images/transfer_learning#\
        # important_note_about_batchnormalization_layers
        # update the statistics of bn
        c3, c4, c5 = self.backbone_resnet(inputs, training=False)
        fpn_out = self.backbone_fpn(c3, c4, c5)

        # Prediction Head branch
        cls_scores = []
        bbox_preds = []
        centernesses = []
        cof_preds = []
        feat_masks = []
        count = 0

        # all output from FPN use same prediction head
        for stride, f_map in zip(self.strides, fpn_out):
            cof_pred, centerness, cls_score, bbox_pred, reg_feat = \
               self.predictionHead(f_map)
            cof_preds.append(cof_pred)
            centernesses.append(centerness)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred*stride)

            if count < 3:
                if count == 0:
                    feat_masks.append(reg_feat)
                else:
                    feat_up = tf.keras.layers.UpSampling2D(
                        size=2 ** count, interpolation='bilinear')(reg_feat)
                    feat_masks.append(feat_up)
            count = count + 1
        
        feat_masks = tf.concat(feat_masks, dim=1)
        feat_masks = tf.nn.relu(
            self.sip_mask_lat(tf.nn.relu(self.sip_mask_lat0(feat_masks))))
        feat_masks = tf.keras.layers.UpSampling2D(
            size=4, interpolation='bilinear')(feat_masks)
 
        cls_scores = tf.concat(cls_scores, axis=1)
        bbox_preds = tf.concat(bbox_preds, axis=1)
        centernesses = tf.concat(centernesses, axis=1)
        cof_preds = tf.concat(cof_preds, axis=1)

        if training:
            pred = {
                'cls_scores': cls_scores,
                'bbox_preds': bbox_preds,
                'centernesses': centernesses,
                'cof_preds': cof_preds,
                'feat_masks': feat_masks,
            }
            # Following to make both `if` and `else` return structure same
            result = {
                'detection_boxes': tf.zeros((self.max_output_size, 4)),
                'detection_classes': tf.zeros((self.max_output_size)), 
                'detection_scores': tf.zeros((self.max_output_size)), 
                'detection_masks': tf.zeros((self.max_output_size, 30, 30, 1)), 
                'num_detections': tf.constant(0)}
            pred.update(result)
        else:
            pred = {
                'cls_scores': cls_scores,
                'bbox_preds': bbox_preds,
                'centernesses': centernesses,
                'cof_preds': cof_preds,
                'feat_masks': feat_masks,
            }

            pred.update(self.detect(pred, img_shape=tf.shape(inputs)))

        return pred
