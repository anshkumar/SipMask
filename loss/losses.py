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
                 max_masks_for_train=100):
        self.img_h = img_h
        self.img_w = img_w
        self._loss_weight_cls = loss_weight_cls
        self._loss_weight_box = loss_weight_box
        self._loss_weight_mask = loss_weight_mask
        self._loss_weight_mask_iou = loss_weight_mask_iou
        self._loss_weight_seg = loss_seg
        self._neg_pos_ratio = neg_pos_ratio
        self._max_masks_for_train = max_masks_for_train
        self.center_sampling = True
        self.center_sample_radius = 1.5
        self.strides = [4, 8, 16, 32, 64]
        self.regress_ranges = (
            (-1.0, 64.0), 
            (64.0, 128.0), 
            (128.0, 256.0), 
            (256.0, 512.0), 
            (512.0, INF))

    def __call__(self, model, pred, label, num_classes, image = None):
        """
        Args:
            num_classes: Num_classes + 1 for background
            anchors:
            label: labels dict from dataset
                all_offsets: the transformed box coordinate offsets of each pair of 
                          prior and gt box
                conf_gt: the foreground and background labels according to the 
                         'pos_thre' and 'neg_thre',
                         '0' means background, '>0' means foreground.
                prior_max_box: the corresponding max IoU gt box for each prior
                prior_max_index: the index of the corresponding max IoU gt box for 
                          each prior
            pred:
        Returns:
        """
        self.image = image
        # all prediction component
        cls_scores = pred['cls_scores']
        bbox_preds = pred['bbox_preds']
        centernesses = pred['centernesses']
        cof_preds = pred['cof_preds']
        feat_masks = pred['feat_masks']

        # all label component
        gt_bboxes = label['boxes'] # [ymin, xmin, ymax, xmax ]
        # Change to [xmin, ymin, xmax , ymax ] format
        gt_bboxes = tf.stack((gt_bboxes[..., 1], gt_bboxes[..., 0], 
            gt_bboxes[..., 3], gt_bboxes[..., 2]), -1)
        gt_masks = label['mask_target']
        gt_labels = tf.cast(label['classes'], tf.int32)
        self.num_classes = num_classes - 1
        self.model = model

        all_level_points, all_level_strides = self.get_points(
            self.model.feature_map_size, tf.float32)
                
        labels, bbox_targets, label_list, bbox_targets_list, gt_inds = \
            self.fcos_target(all_level_points, gt_bboxes, gt_labels)
        

        #decode detection and groundtruth
        det_bboxes = []
        det_targets = []
        num_levels = len(bbox_preds)
        batch_size = tf.shape(label['boxes'])[0]

        for img_id in range(batch_size):            
            bbox_pred_list = [
                tf.reshape(bbox_preds[i][img_id], (-1, 4)) for i in range(
                    num_levels)
            ]
            bbox_target_list =  bbox_targets_list[img_id]

            bboxes = []
            targets = []
            for i in range(len(bbox_pred_list)):
                bbox_pred = bbox_pred_list[i]
                bbox_target = bbox_target_list[i]
                points = all_level_points[i]
                bboxes.append(self._distance2bbox(points, bbox_pred))
                targets.append(self._distance2bbox(points, bbox_target))

            bboxes = tf.concat (bboxes, axis=0)
            targets = tf.concat (targets, axis=0)

            det_bboxes.append(bboxes)
            det_targets.append(targets)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            tf.reshape(cls_score, (-1, tf.shape(cls_score)[-1]))
            for cls_score in cls_scores
        ]

        flatten_bbox_preds = [
            tf.reshape(bbox_pred, (-1, 4))
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            tf.reshape(centerness, (-1))
            for centerness in centernesses
        ]

        flatten_cls_scores = tf.concat(flatten_cls_scores, axis=0)
        flatten_bbox_preds = tf.concat(flatten_bbox_preds, axis=0)
        flatten_centerness = tf.concat(flatten_centerness, axis=0)
        flatten_labels = tf.concat(labels, axis=0)
        flatten_bbox_targets = tf.concat(bbox_targets, axis=0)

        # repeat points to align with bbox_preds
        flatten_points = tf.concat([tf.tile(points, (batch_size,1)) \
            for points in all_level_points], axis=0)
        flatten_strides = tf.concat([tf.tile(tf.reshape(strides, (-1,1)), 
            (batch_size, 1)) for strides in all_level_strides], axis=0)

        pos_inds = tf.where(flatten_labels > 0)
        num_pos = tf.shape(pos_inds)[0]

        loss_cls = self._focal_conf_sigmoid_loss(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + batch_size)

        pos_bbox_preds = tf.gather_nd(flatten_bbox_preds, pos_inds)
        pos_centerness = tf.gather_nd(flatten_centerness, pos_inds)

        if num_pos > 0:
            pos_bbox_targets = tf.gather_nd(flatten_bbox_targets, pos_inds)
            pos_centerness_targets = self._centerness_target(pos_bbox_targets)

            pos_points = tf.gather_nd(flatten_points, pos_inds)
            pos_strides = tf.gather_nd(flatten_strides, pos_inds)

            pos_decoded_bbox_preds = self._distance2bbox(pos_points, 
                pos_bbox_preds/pos_strides)
            pos_decoded_target_preds = self._distance2bbox(pos_points,
                pos_bbox_targets/pos_strides)

            # centerness weighted iou loss
            loss_bbox = self._loss_location(pos_decoded_target_preds, 
                pos_decoded_bbox_preds, pos_centerness_targets) / \
                tf.reduce_sum(pos_centerness_targets)

            loss_centerness = self._loss_centerness(pos_centerness,
                pos_centerness_targets)
        else:
            loss_bbox = tf.reduce_sum(pos_bbox_preds)
            loss_centerness = tf.reduce_sum(pos_centerness)
            
        import pdb
        pdb.set_trace()

    def _loss_location(self, y_true, y_pred, sample_weight=None):
        gl = tfa.losses.GIoULoss()
        return gl(y_true, y_pred, sample_weight)

    def _loss_centerness(self, y_true, y_pred, weight=1.0):
        loss_centerness = tf.nn.sigmoid_cross_entropy_with_logits(y_true,
            y_pred)*weight
        return tf.reduce_mean(loss_centerness)

    def _focal_conf_sigmoid_loss(self, flatten_cls_scores, flatten_labels,
        avg_factor, focal_loss_alpha=0.25, focal_loss_gamma=2):
        """
        Focal loss but using sigmoid like the original paper.
        """
        labels = tf.one_hot(flatten_labels, depth=self.num_classes)

        fl = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, 
            reduction=tf.keras.losses.Reduction.SUM)
        loss = fl(y_true=labels, y_pred=flatten_cls_scores)

        return tf.math.divide_no_nan(loss, tf.cast(avg_factor, tf.float32))

    def _loss_mask(self, use_cropped_mask=True):
        pass

    def _mask_iou(self, mask1, mask2):
        intersection = tf.reduce_sum(mask1*mask2, axis=(0, 1))
        area1 = tf.reduce_sum(mask1, axis=(0, 1))
        area2 = tf.reduce_sum(mask2, axis=(0, 1))
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def _distance2bbox(self, points, distance):
        """
        Decode distance prediction to bounding box.

        Args:
            points (Tensor): Shape (B, N, 2) or (N, 2).
            distance (Tensor): Distance from the given point to 4
                boundaries (left, top, right, bottom). Shape (B, N, 4) or (N, 4)

        Returns:
            Tensor: Boxes with shape (N, 4) or (B, N, 4)
        """

        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]

        bboxes = tf.stack([x1, y1, x2, y2], -1)
        return bboxes

    def _centerness_target(self, pos_bbox_targets):
        """
        The center-ness depicts the normalized distance from the location to the
        center of the object that the location is responsible for. Given the 
        regression targets l∗, t∗, r∗ and b∗ for a location, the center-ness 
        target is defined as, 
                sqrt( min(l∗, r∗)/max(l∗, r∗) * min(t∗, b∗)/max(t∗, b∗))

        We employ sqrt here to slow down the decay of the centerness. 
        The center-ness ranges from 0 to 1 and is thus trained with binary 
        cross entropy (BCE) loss.

        Args:
            pos_bbox_targets: Tensor of shape [Num_pos, 4]. Positive target 
                                bounding boxes.

        Returns:
            centerness_targets: Tensor of shape [Num_pos, 4]. Positive target 
                                Centerness.
        """

        # only calculate pos centerness targets, otherwise there may be nan
        left_right = tf.stack((pos_bbox_targets[:,0], pos_bbox_targets[:,2]),-1)
        top_bottom = tf.stack((pos_bbox_targets[:,1], pos_bbox_targets[:,3]),-1)
        
        centerness_targets = \
            tf.reduce_min(left_right, -1) / tf.reduce_max(left_right, -1) * \
            tf.reduce_min(top_bottom, -1) / tf.reduce_max(top_bottom, -1)

        return tf.sqrt(centerness_targets)

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
                    [[self.strides[lvl_idx] * radius]], 
                    [lvl_end-lvl_begin, num_gts])
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
        indices = tf.stack(
            (tf.range(num_points, dtype=tf.int64), min_area_inds), axis=-1)
        bbox_targets = tf.gather_nd(bbox_targets, indices)

        indices = tf.where(labels > 0)
        gt_ind = tf.gather_nd(min_area_inds, indices)

        return labels, bbox_targets, gt_ind

    def get_points(self, featmap_sizes, dtype):
        """
        Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(tf.shape(featmap_sizes)[0]):
            points, strides = self.get_points_single(
                featmap_sizes[i], self.strides[i], dtype)
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
