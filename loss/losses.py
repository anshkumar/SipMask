import tensorflow as tf
import tensorflow_addons as tfa
import time

INF = 1e8
class SipMaskLoss(object):
    def __init__(self, 
                 img_h,
                 img_w, 
                 loss_weight_cls=1.0,
                 loss_weight_box=1.5,
                 loss_weight_mask=6.125,
                 loss_weight_mask_iou=25.0,
                 loss_seg=1.0,
                 neg_pos_ratio=3,
                 max_masks_for_train=100, 
                 use_mask_iou=False):
        self.img_h = img_h
        self.img_w = img_w
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_mask_iou = loss_weight_mask_iou
        self._loss_weight_seg = loss_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train
        self.use_mask_iou = use_mask_iou

    def __call__(self, model, pred, label, num_classes, image = None):
        """
        :param num_classes:
        :param anchors:
        :param label: labels dict from dataset
            all_offsets: the transformed box coordinate offsets of each pair of 
                      prior and gt box
            conf_gt: the foreground and background labels according to the 
                     'pos_thre' and 'neg_thre',
                     '0' means background, '>0' means foreground.
            prior_max_box: the corresponding max IoU gt box for each prior
            prior_max_index: the index of the corresponding max IoU gt box for 
                      each prior
        :param pred:
        :return:
        """
        self.image = image
        # all prediction component
        self.cls_scores = pred['cls_scores']
        self.bbox_preds = pred['bbox_preds']
        self.centernesses = pred['centernesses']
        self.cof_preds = pred['cof_preds']
        self.feat_masks = pred['feat_masks']

        # all label component
        self.gt_offset = label['all_offsets']
        self.conf_gt = label['conf_gt']
        self.prior_max_box = label['prior_max_box']
        self.prior_max_index = label['prior_max_index']

        self.masks = label['mask_target']
        self.classes = label['classes']
        self.num_classes = num_classes
        self.model = model

        # TODO: check the feature map size
        featmap_sizes = [featmap.size()[1:-1] for featmap in cls_scores]
        all_level_points, all_level_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype)

        loc_loss = self._loss_location() 

        conf_loss = self._loss_class_ohem() 

        mask_loss, mask_iou_loss = self._loss_mask() 
        mask_iou_loss *= self._loss_weight_mask_iou

        seg_loss = self._loss_semantic_segmentation() 

        total_loss = loc_loss + conf_loss + mask_loss + seg_loss + mask_iou_loss
        
        return loc_loss, conf_loss, mask_loss, mask_iou_loss, seg_loss, \
                total_loss

    def _loss_location(self):
        # only compute losses from positive samples
        # get postive indices
        pos_indices = tf.where(self.conf_gt > 0 )
        pred_offset = tf.gather_nd(self.pred_offset, pos_indices)
        gt_offset = tf.gather_nd(self.gt_offset, pos_indices)

        # calculate the smoothL1(positive_pred, positive_gt) and return
        num_pos = tf.shape(gt_offset)[0]
        smoothl1loss = tf.keras.losses.Huber(delta=1.)
        if tf.reduce_sum(tf.cast(num_pos, tf.float32)) > 0.0:
            loss_loc = smoothl1loss(gt_offset, pred_offset)
        else:
            loss_loc = 0.0

        tf.debugging.assert_all_finite(loss_loc, "Loss Location NaN/Inf")

        return loss_loc*self._loss_weight_box

    def _focal_conf_sigmoid_loss(self, focal_loss_alpha=0.75, focal_loss_gamma=2):
        """
        Focal loss but using sigmoid like the original paper.
        """
        labels = tf.one_hot(self.conf_gt, depth=num_cls)
        # filter out "neutral" anchors
        indices = tf.where(self.conf_gt >= 0)
        labels = tf.gather_nd(labels, indices)
        pred_cls = tf.gather_nd(self.pred_cls, indices)

        fl = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, 
            reduction=tf.keras.losses.Reduction.SUM)
        loss = fl(y_true=labels, y_pred=pred_cls)

        pos_indices = tf.where(self.conf_gt > 0 )
        num_pos = tf.shape(pos_indices)[0]
        return loss #tf.math.divide_no_nan(loss, tf.cast(num_pos, tf.float32))

    def _loss_class(self):
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)

        loss_conf = scce(tf.cast(self.conf_gt, dtype=tf.int32), self.pred_cls, 
                            self._loss_weight_cls)

        return loss_conf

    def _loss_class_ohem(self):
        # num_cls includes background
        batch_conf = tf.reshape(self.pred_cls, [-1, self.num_classes])

        # Hard Negative Mining
        # Using tf.nn.softmax or tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1)) to calculate log_sum_exp
        # might cause NaN problem. This is a known problem https://github.com/tensorflow/tensorflow/issues/10142
        # To get around this using tf.math.reduce_logsumexp and softmax_cross_entropy_with_logit

        # This will be used to determine unaveraged confidence loss across all examples in a batch.
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/modules/multibox_loss.py#L251
        # https://github.com/dbolya/yolact/blob/b97e82d809e5e69dc628930070a44442fd23617a/layers/box_utils.py#L316
        # log_sum_exp = tf.math.log(tf.math.reduce_sum(tf.math.exp(batch_conf), 1))

        # Using inbuild reduce_logsumexp to avoid NaN
        # This function is more numerically stable than log(sum(exp(input))). It avoids overflows caused by taking the exp of large inputs and underflows caused by taking the log of small inputs.
        log_sum_exp = tf.math.reduce_logsumexp(batch_conf, 1)
        # tf.print(log_sum_exp)
        loss_c = log_sum_exp - batch_conf[:,0]

        loss_c = tf.reshape(loss_c, (tf.shape(self.pred_cls)[0], -1))  # (batch_size, 27429)
        pos_indices = tf.where(self.conf_gt > 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, pos_indices, tf.zeros(tf.shape(pos_indices)[0])) # filter out pos boxes
        num_pos = tf.math.count_nonzero(tf.greater(self.conf_gt,0), axis=1, keepdims=True)
        num_neg = tf.clip_by_value(num_pos * self._neg_pos_ratio, clip_value_min=tf.constant(self._neg_pos_ratio, dtype=tf.int64), clip_value_max=tf.cast(tf.shape(self.conf_gt)[1]-1, tf.int64))

        neutrals_indices = tf.where(self.conf_gt < 0 )
        loss_c = tf.tensor_scatter_nd_update(loss_c, neutrals_indices, tf.zeros(tf.shape(neutrals_indices)[0])) # filter out neutrals (conf_gt = -1)

        idx = tf.argsort(loss_c, axis=1, direction='DESCENDING')
        idx_rank = tf.argsort(idx, axis=1)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        # Filter out neutrals and positive
        neg_indices = tf.where((tf.cast(idx_rank, dtype=tf.int64) < num_neg) & (self.conf_gt == 0))

        # neg_indices shape is (batch_size, no_prior)
        # pred_cls shape is (batch_size, no_prior, no_class)
        neg_pred_cls_for_loss = tf.gather_nd(self.pred_cls, neg_indices)
        neg_gt_for_loss = tf.gather_nd(self.conf_gt, neg_indices)
        pos_pred_cls_for_loss = tf.gather_nd(self.pred_cls, pos_indices)
        pos_gt_for_loss = tf.gather_nd(self.conf_gt, pos_indices)

        target_logits = tf.concat([pos_pred_cls_for_loss, neg_pred_cls_for_loss], axis=0)
        target_labels = tf.concat([pos_gt_for_loss, neg_gt_for_loss], axis=0)
        target_labels = tf.one_hot(tf.squeeze(target_labels), depth=self.num_classes)

        if tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32)) > 0.0:
            cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                reduction=tf.keras.losses.Reduction.SUM)
            loss_conf = cce(target_labels, target_logits) / tf.reduce_sum(tf.cast(num_pos, tf.float32)+tf.cast(num_neg, tf.float32))
        else:
            loss_conf = 0.0
        return loss_conf*self._loss_weight_cls

    def _loss_mask(self, use_cropped_mask=True):

        shape_proto = tf.shape(self.proto_out)
        proto_h = shape_proto[1]
        proto_w = shape_proto[2]
        num_batch = shape_proto[0]
        loss_m = 0.0
        loss_iou = 0.0

        #[batch, height, width, num_object]
        mask_gt = tf.transpose(self.masks, (0,2,3,1)) 

        maskiou_t_list = []
        maskiou_net_input_list = []
        class_t_list = []
        total_pos = 0

        for i in tf.range(num_batch):
            pos_indices = tf.where(self.conf_gt[i] > 0 )

            #shape: [num_positives]
            _pos_prior_index = tf.gather_nd(self.prior_max_index[i], pos_indices) 

            #shape: [num_positives]
            _pos_prior_box = tf.gather_nd(self.prior_max_box[i], pos_indices) 

            #shape: [num_positives]
            _pos_coef = tf.gather_nd(self.pred_mask_coef[i], pos_indices)

            _mask_gt = mask_gt[i]
            cur_class_gt = self.classes[i]

            if tf.shape(_pos_prior_index)[0] == 0: # num_positives are zero
                continue
            
            # If exceeds the number of masks for training, 
            # select a random subset
            old_num_pos = tf.shape(_pos_coef)[0]
            
            if old_num_pos > self._max_masks_for_train:
                perm = tf.random.shuffle(tf.range(tf.shape(_pos_coef)[0]))
                select = perm[:self._max_masks_for_train]
                _pos_coef = tf.gather(_pos_coef, select)
                _pos_prior_index = tf.gather(_pos_prior_index, select)
                _pos_prior_box = tf.gather(_pos_prior_box, select)

            num_pos = tf.shape(_pos_coef)[0]
            total_pos += num_pos
            pos_mask_gt = tf.gather(_mask_gt, _pos_prior_index, axis=-1) 
            pos_class_gt = tf.gather(cur_class_gt, _pos_prior_index, axis=-1)   
            
            # mask assembly by linear combination
            mask_p = tf.linalg.matmul(self.proto_out[i], _pos_coef, transpose_a=False, 
                transpose_b=True) # [proto_height, proto_width, num_pos]
            mask_p = tf.sigmoid(mask_p)

            # crop the pred (not real crop, zero out the area outside the 
            # gt box)
            if use_cropped_mask:
                # _pos_prior_box.shape: (num_pos, 4)
                # bboxes_for_cropping = tf.stack([
                #     _pos_prior_box[:, 0]/self.img_h, 
                #     _pos_prior_box[:, 1]/self.img_w,
                #     _pos_prior_box[:, 2]/self.img_h,
                #     _pos_prior_box[:, 3]/self.img_w
                #     ], axis=-1)
                # mask_p = utils.crop(mask_p, bboxes_for_cropping)
                mask_p = utils.crop(mask_p, _pos_prior_box)  
                # pos_mask_gt = utils.crop(pos_mask_gt, _pos_prior_box)

            # mask_p = tf.clip_by_value(mask_p, clip_value_min=0.0, 
            #     clip_value_max=1.0)

            # Divide the loss by normalized boxes width and height to get 
            # ROIAlign affect. 

            # Getting normalized boxes widths and height
            # boxes_w = (_pos_prior_box[:, 3] - _pos_prior_box[:, 1])/self.img_w
            # boxes_h = (_pos_prior_box[:, 2] - _pos_prior_box[:, 0])/self.img_h
            boxes_w = (_pos_prior_box[:, 3] - _pos_prior_box[:, 1])
            boxes_h = (_pos_prior_box[:, 2] - _pos_prior_box[:, 0])

            # Adding extra dimension as i/p and o/p shapes are different with 
            # "reduction" is set to None.
            # https://github.com/tensorflow/tensorflow/issues/27190
            _pos_mask_gt = tf.transpose(pos_mask_gt, (2,0,1))
            _mask_p = tf.transpose(mask_p, (2,0,1))
            _pos_mask_gt = tf.expand_dims(_pos_mask_gt, axis=-1)
            _mask_p = tf.expand_dims(_mask_p, axis=-1)
                       
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, 
                reduction=tf.losses.Reduction.NONE)
            mask_loss = bce(_pos_mask_gt, _mask_p)

            mask_loss = tf.reduce_mean(mask_loss, 
                                        axis=(1,2)) 

            tf.debugging.assert_all_finite(mask_loss, "Mask Loss NaN/Inf")

            if use_cropped_mask:
                mask_loss = tf.math.divide_no_nan(mask_loss, boxes_w * boxes_h)
            
            mask_loss = tf.reduce_sum(mask_loss)
            
            if old_num_pos > num_pos:
                mask_loss *= tf.cast(old_num_pos / num_pos, tf.float32)

            loss_m += mask_loss

            # Mask IOU loss
            if self.use_mask_iou:
                pos_mask_gt_area = tf.reduce_sum(pos_mask_gt, axis=(0,1))

                # Area threshold of 25 pixels
                select_indices = tf.where(pos_mask_gt_area > 25 ) 

                if tf.shape(select_indices)[0] == 0: # num_positives are zero
                    continue

                _pos_prior_box = tf.gather_nd(_pos_prior_box, select_indices)
                mask_p = tf.gather(mask_p, tf.squeeze(select_indices), axis=-1)
                pos_mask_gt = tf.gather(pos_mask_gt, tf.squeeze(select_indices), 
                    axis=-1)
                pos_class_gt = tf.gather_nd(pos_class_gt, select_indices)

                mask_p = tf.cast(mask_p + 0.5, tf.uint8)
                mask_p = tf.cast(mask_p, tf.float32)
                maskiou_t = self._mask_iou(mask_p, pos_mask_gt)

                if tf.size(maskiou_t) == 1:
                    maskiou_t = tf.expand_dims(maskiou_t, axis=0)
                    mask_p = tf.expand_dims(mask_p, axis=-1)

                maskiou_net_input_list.append(mask_p)
                maskiou_t_list.append(maskiou_t)
                class_t_list.append(pos_class_gt)

        loss_m = tf.math.divide_no_nan(loss_m, tf.cast(total_pos, tf.float32))

        if len(maskiou_t_list) == 0:
            return loss_m , loss_iou
        else:
            maskiou_t = tf.concat(maskiou_t_list, axis=0)
            class_t = tf.concat(class_t_list, axis=0)
            maskiou_net_input = tf.concat(maskiou_net_input_list, axis=-1)

            maskiou_net_input = tf.transpose(maskiou_net_input, (2,0,1))
            maskiou_net_input = tf.expand_dims(maskiou_net_input, axis=-1)
            num_samples = tf.shape(maskiou_t)[0]
            # TODO: train random sample (maskious_to_train)

            maskiou_p = self.model.fastMaskIoUNet(maskiou_net_input)

            # Using index zero for class label.
            # Indices are K-dimensional. 
            # [number_of_selections, [1st_dim_selection, 2nd_dim_selection, ..., 
            #  kth_dim_selection]]
            indices = tf.concat(
                (
                    tf.expand_dims(tf.range((num_samples), 
                        dtype=tf.int64), axis=-1), 
                    tf.expand_dims(class_t-1, axis=-1)
                ), axis=-1)
            maskiou_p = tf.gather_nd(maskiou_p, indices)

            smoothl1loss = tf.keras.losses.Huber(delta=1.)
            loss_i = smoothl1loss(maskiou_t, maskiou_p)

            loss_iou += loss_i

        return loss_m*self._loss_weight_mask , loss_iou

    def _mask_iou(self, mask1, mask2):
        intersection = tf.reduce_sum(mask1*mask2, axis=(0, 1))
        area1 = tf.reduce_sum(mask1, axis=(0, 1))
        area2 = tf.reduce_sum(mask2, axis=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        '''
        Function calculates bouding box targets for a batch of images by 
        calling fcos_target_single for every single image.

        Args:
            points: List of shape [Num_level]. Each entry is of shape 
                    [num_point, 2]. Points for evry level of fpn.
            gt_bboxes_list: Tensor of shape [num_images_batch, num_bboxes, 4].
                            All the ground truth bounding boxes for a batch.
            gt_labels_list: Tensor of shape [num_images_batch, num_bboxes].
                            Label of ground truth bounding boxes for a batch.

        Returns:
            concat_lvl_labels: List of shape [Num_level]. Each entry is of shape 
                            [num_point_level*num_images_batch].
                            labels concatenated to per level for a batch of 
                            images.
            concat_lvl_bbox_targets: List of shape [Num_level]. Each entry is of
                            shape [num_point_level*num_images_batch, 4]. 
                            bbox_targets concatenated to per level for a batch 
                            of images.
            labels_list:  List of shape [num_images_batch]. Each entry is of
                            shape [num_point_level]. labels list splited to per 
                            image, per level.
            bbox_targets_list: List of shape [num_images_batch]. Each entry is 
                            of shape [num_point_level, 4]. bbox targets splited 
                            to per image, per level.
            gt_inds_list: List of shape [num_images_batch]. indicies of points 
                            matching with the ground truth for a batch of 
                            images.

        '''

        num_levels = len(points)
        expanded_regress_ranges = [tf.tile(
            tf.expand_dims(self.regress_ranges[i], axis=0),
            (tf.shape(points[i])[0], 1)) \
            for i in range(len(points))]
        concat_regress_ranges = tf.concat(expanded_regress_ranges, axis=0)

        concat_points =  tf.concat(points, axis=0)

        num_points = [tf.shape(center)[0] \
                        for center in points]

        labels_list = []
        bbox_targets_list = []
        gt_inds_list = []
        for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
            labels, bbox_targets, gt_inds = self.fcos_target_single(gt_bboxes, 
                gt_labels, concat_points, concat_regress_ranges, num_points)

            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)
            gt_inds_list.append(gt_inds)

        # split to per img, per level
        labels_list = [
            tf.split(labels, num_points, axis=0) for labels in labels_list]

        bbox_targets_list = [
            tf.split(bbox_targets, num_points, axis=0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                tf.concat([labels[i] for labels in labels_list], axis=0))
            concat_lvl_bbox_targets.append(
                tf.concat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list], 
                    axis=0))
        
        return concat_lvl_labels, concat_lvl_bbox_targets, labels_list, \
                bbox_targets_list, gt_inds_list


    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        '''
        Function calculates bouding box targets and filters accoring to the 
        following conditions:
            1) location (x,y) is considered as a positive sample if it falls 
                into any ground-truth box and the class label c∗ of the 
                location is the class label of the ground-truth box. Otherwise 
                it is a negative sample and c∗ = 0 (back- ground class).
            2) We firstly compute the regression targets l∗, t∗, r∗ and b∗ for 
                each location on all feature levels. Next, if a location 
                satisfies max(l∗, t∗, r∗, b∗) > mi or max(l∗, t∗,r∗,b∗) < mi−1, 
                it is set as a negative sample and is thus not required to 
                regress a bounding box anymore. Here mi is the maximum 
                distance that feature level i needs to regress. In this work, 
                m2, m3, m4, m5, m6 and m7 are set as 0, 64, 128, 256, 512 and 
                inf, respectively.
            3) If a location falls into multiple bounding boxes, it is 
                considered as an ambiguous sample. We simply choose the bounding
                box with minimal area as its regression target 
        If center_sampling is True then fcos_plus logic is used as in
            https://github.com/yqyao/FCOS_PLUS/blob/master/fcos.pdf

        Args:
            gt_bboxes: Tensor of shape [num_gt_boxes_per_image, 4]. All the 
                        ground truth bounding boxes per image.
            gt_labels: Tensor of shape [num_gt_boxes_per_image]. Label of 
                        ground truth bounding boxes.
            points: Tensor of shape [num_points, 2]. (x,y) location of points 
                        with repect to the original image.
            regress_ranges: Tensor of shape [num_points, 2]. Max distance 
                        (pixels) to search for a point in a bounding box.
            num_points_per_lvl: List of shape [num_features_level]. Number of 
                        points to take from each feature level.
        Returns:
            labels: Label for each point from the feature map. 0 for background.
            bbox_targets: bbox target for each point from the feature map.
            gt_ind: indicies of points matching with the ground truth.
        '''
        num_points = tf.shape(points)[0]
        num_gts = tf.shape(gt_labels)[0]

        if num_gts == 0:
            return tf.zeros(num_points, dtype=tf.int32), \
                   tf.zeros((num_points, 4), dtype=tf.float32)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        areas =  tf.tile(tf.expand_dims(areas, axis=0), (num_points, 1))
        regress_ranges = tf.tile(
            tf.expand_dims(regress_ranges, axis=1), (1, num_gts, 1))
        gt_bboxes =  tf.tile(
            tf.expand_dims(gt_bboxes, axis=0), (num_points, 1, 1))

        xs, ys = points[:, 0], points[:, 1]
        xs = tf.tile(tf.expand_dims(xs, axis=1), (1, num_gts))
        ys = tf.tile(tf.expand_dims(ys, axis=1), (1, num_gts))

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = tf.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            stride = tf.zeros_like(center_xs)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0

            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl

                # slice and update as follows:  
                #    stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                indices = tf.expand_dims(tf.range(lvl_begin,lvl_end), axis=-1)
                updates = tf.tile(
                    [[self.strides[lvl_idx] * radius]], [lvl_end-lvl_begin, num_gts])
                stride = tf.tensor_scatter_nd_update(stride, indices, updates)

                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts_0 = tf.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts_1 = tf.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts_2 = tf.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts_3 = tf.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)
            center_gts = tf.stack(
                (center_gts_0, center_gts_1, center_gts_2, center_gts_3), 
                axis=-1)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = tf.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = tf.math.reduce_min(center_bbox, axis=-1) > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = tf.math.reduce_min(bbox_targets, axis=-1) > 0

        # condition2: limit the regression range for each location
        max_regress_distance = tf.math.reduce_max(bbox_targets, axis=-1)
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area

        # equivalent to:  areas[inside_gt_bbox_mask == 0] = INF
        indices = tf.where(inside_gt_bbox_mask == False)
        updates = tf.repeat(INF, tf.shape(indices)[0])
        areas = tf.tensor_scatter_nd_update(areas, indices, updates)

        # equivalent to: areas[inside_regress_range == 0] = INF
        indices = tf.where(inside_regress_range == False)
        updates = tf.repeat(INF, tf.shape(indices)[0])
        areas = tf.tensor_scatter_nd_update(areas, indices, updates)

        min_area = tf.reduce_min(areas, axis=1)
        min_area_inds = tf.argmin(areas, axis=1)

        # Equivalent to gt_labels[min_area_inds]
        labels = tf.gather_nd(gt_labels, tf.expand_dims(min_area_inds, axis=-1))

        # Equivalent to labels[min_area == INF] = 0
        indices = tf.where(min_area == INF)
        updates = tf.repeat(0, tf.shape(indices)[0])
        labels = tf.tensor_scatter_nd_update(labels, indices, updates)

        # Equivalent to bbox_targets[range(num_points), min_area_inds]
        indices = tf.stack((tf.range(num_points, dtype=tf.int64), min_area_inds), axis=-1)
        bbox_targets = tf.gather_nd(bbox_targets, indices)

        indices = tf.where(labels > 0)
        gt_ind = tf.gather_nd(min_area_inds, indices)

        return labels, bbox_targets, gt_ind

    def get_points(self, featmap_sizes, dtype):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(tf.shape(featmap_sizes)[0]):
            points, strides = self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single(self, featmap_size, stride, dtype):
        h, w = featmap_size
        x_range = tf.range(
            0, w * stride, stride, dtype=dtype)
        y_range = tf.range(
            0, h * stride, stride, dtype=dtype)
        y, x = tf.meshgrid(y_range, x_range, indexing='ij')
        points = tf.stack(
            (tf.reshape(x, (-1)), tf.reshape(y, (-1))), axis=-1) + stride // 2
        strides = points[:,0]*0+stride
        return points, strides
