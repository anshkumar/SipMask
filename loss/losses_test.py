from absl.testing import parameterized
import tensorflow as tf
from losses import SipMaskLoss
import torch

INF = 1e8
class SipMaskLossTest(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.loss = SipMaskLoss(550,550)
        self.loss.center_sampling = True
        self.loss.center_sample_radius = 1.5
        self.loss.strides = [4, 8, 16, 32, 64]
        self.strides = self.loss.strides
        self.center_sample_radius = self.loss.center_sample_radius
        self.featmap_sizes = [[100, 168], [50, 84], [25, 42], [13, 21], [7, 11]]
        self.regress_ranges = (
            (-1.0, 64.0), 
            (64.0, 128.0), 
            (128.0, 256.0), 
            (256.0, 512.0), 
            (512.0, INF))
        self.loss.regress_ranges = self.regress_ranges

    def fcos_target_pytorch(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list = []
        bbox_targets_list = []
        gt_inds = []
        for gt_bboxes, gt_labels in zip(gt_bboxes_list, gt_labels_list):
            labels, bbox_targets, gt_ind = self.fcos_target_single_pytorch(
                gt_bboxes, gt_labels, concat_points, concat_regress_ranges, 
                num_points)

            labels_list.append(labels)
            bbox_targets_list.append(bbox_targets)
            gt_inds.append(gt_ind)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, labels_list, \
                bbox_targets_list, gt_inds

    def fcos_target_single_pytorch(self, gt_bboxes, gt_labels, points, 
        regress_ranges, num_points_per_lvl):

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        strides = self.strides

        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        bbox_targets = bbox_targets

        if True:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_ind = min_area_inds[labels > 0]

        return labels, bbox_targets, gt_ind

    def get_points_pytorch(self, featmap_sizes, dtype):
        """Get points according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.
        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single_pytorch(featmap_sizes[i], 
                self.strides[i], dtype)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single_pytorch(self, featmap_size, stride, dtype):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        strides = points[:,0]*0+stride
        return points, strides

    def centerness_target_pytorch(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    # @parameterized.parameters(
    #         ([[-0.96260047,  0.5351808 ,  0.11765705, -0.65827477,
    #              0.6153985 , -0.33301416],
    #            [-1.9649765 ,  0.6486603 ,  0.6216589 , -0.9070863 ,
    #             -0.28419286, -0.11695653],
    #            [-1.9652086 ,  0.61087036,  0.71910125, -0.31642425,
    #              0.02182993, -0.19831532]], 
    #         [0, 4, 0],
    #         )
    #     )
    # def test_focal_conf_sigmoid_loss(self, flatten_cls_scores, flatten_labels,
    #     avg_factor):

    #     self.loss.num_classes = 6
    #     loss_cls = self.loss._focal_conf_sigmoid_loss(flatten_cls_scores, 
    #         flatten_labels, 3)

    @parameterized.parameters(
            ([[  1.6697693 ,   9.82251   ,   8.641907  ,  21.211517  ],
               [  5.6697693 ,   9.82251   ,   4.6419067 ,  21.211517  ],
               [  9.669769  ,   9.82251   ,   0.64190674,  21.211517  ],
               [  1.6697693 ,  13.82251   ,   8.641907  ,  17.211517  ]], )
        )
    def test_centerness_target(self, pos_bbox_targets):
        centerness = self.loss._centerness_target(tf.constant(pos_bbox_targets))
        centerness_targets = self.centerness_target_pytorch(
                torch.tensor(pos_bbox_targets))

        self.assertAllClose(centerness, centerness_targets)

    @parameterized.parameters(
            ([[100, 168], [50, 84], [25, 42], [13, 21], [7, 11]], )
        )
    def test_get_points(self, featmap_sizes):
        all_level_points, all_level_strides = self.loss.get_points(
            tf.constant(featmap_sizes), tf.float32)     
        all_level_points_t, all_level_strides_t = self.get_points_pytorch(
            featmap_sizes, torch.float32)
        self.assertAllClose(all_level_points, all_level_points_t)
        self.assertAllClose(all_level_strides, all_level_strides_t)

    @parameterized.parameters(
            (tf.constant([
                        [[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]],
                        [[5.0, 5.0, 10.0, 10.0], [15.0, 15.0, 20.0, 20.0]]
                        ]), 
            tf.constant([[33,  1], [2,  14]])),
        )
    def test_fcos_target(self, gt_bboxes, gt_labels):
        all_level_points, all_level_strides = self.loss.get_points(
            tf.constant(self.featmap_sizes), tf.float32)
        concat_lvl_labels, concat_lvl_bbox_targets, labels_list, \
                bbox_targets_list, gt_inds_list = self.loss.fcos_target(
                                        all_level_points, gt_bboxes, gt_labels)

        all_level_points, _ = self.get_points_pytorch(self.featmap_sizes, 
                                                        torch.float32)
        concat_lvl_labels_t, concat_lvl_bbox_targets_t, labels_list_t, \
                bbox_targets_list_t, gt_inds_list_t = self.fcos_target_pytorch(
                    all_level_points, torch.tensor(gt_bboxes.numpy()), 
                                            torch.tensor(gt_labels.numpy()))

        self.assertAllClose(concat_lvl_labels, concat_lvl_labels_t)
        self.assertAllClose(concat_lvl_bbox_targets, concat_lvl_bbox_targets_t)
        self.assertAllClose(labels_list, labels_list_t)
        self.assertAllClose(bbox_targets_list, bbox_targets_list_t)
        self.assertAllClose(gt_inds_list, gt_inds_list_t)

    @parameterized.parameters(
            (tf.constant([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]]), 
            tf.constant([33,  1])),
        )
    def test_fcos_target_single(self, gt_bboxes, gt_labels):
        all_level_points, all_level_strides = self.loss.get_points(
            tf.constant(self.featmap_sizes), tf.float32)
        num_points_per_lvl = [tf.shape(center)[0].numpy() \
                                for center in all_level_points]
        points =  tf.concat(all_level_points, axis=0)

        regress_ranges = [tf.tile(
            tf.expand_dims(self.regress_ranges[i], axis=0),
            (tf.shape(all_level_points[i])[0], 1)) \
            for i in range(len(all_level_points))]
        regress_ranges = tf.concat(regress_ranges, axis=0)
        
        labels, bbox_targets, gt_ind = self.loss.fcos_target_single(
            gt_bboxes, gt_labels, points, regress_ranges, num_points_per_lvl)
        labels_t, bbox_targets_t, gt_ind_t = self.fcos_target_single_pytorch(
            torch.tensor(gt_bboxes.numpy()), torch.tensor(gt_labels.numpy()), 
            torch.tensor(points.numpy()), torch.tensor(regress_ranges.numpy()), 
            num_points_per_lvl)

        self.assertAllClose(labels.numpy(), labels_t.numpy())
        self.assertAllClose(bbox_targets.numpy(), bbox_targets_t.numpy())
        self.assertAllClose(gt_ind.numpy(), gt_ind_t.numpy())

if __name__ == '__main__':
  tf.test.main()
