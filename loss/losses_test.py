from absl.testing import parameterized
import tensorflow as tf
from losses import SipMaskLoss
import torch

INF = 1e8
class SipMaskLossTest(tf.test.TestCase, parameterized.TestCase):

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

    @parameterized.parameters(
        (
            [[100, 168], [50, 84], [25, 42], [13, 21], [7, 11]], )
        )
    def test_get_points(self, featmap_sizes):
        loss = SipMaskLoss(550,550)  
        loss.strides = [4, 8, 16, 32, 64] 
        self.strides = loss.strides
        all_level_points, all_level_strides = loss.get_points(
            tf.constant(featmap_sizes), tf.float32)     
        all_level_points_t, all_level_strides_t = self.get_points_pytorch(
            featmap_sizes, torch.float32)
        self.assertAllClose(all_level_points, all_level_points_t)
        self.assertAllClose(all_level_strides, all_level_strides_t)

    @parameterized.parameters(
        (
            tf.constant([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]]), 
            tf.constant([33,  1])),
        )
    def test_fcos_target_single(self, gt_bboxes, gt_labels):
        loss = SipMaskLoss(550,550)
        loss.center_sampling = True
        loss.center_sample_radius = 1.5
        loss.strides = [4, 8, 16, 32, 64]
        self.strides = loss.strides
        self.center_sample_radius = loss.center_sample_radius
        featmap_sizes = [[100, 168], [50, 84], [25, 42], [13, 21], [7, 11]]

        all_level_points, all_level_strides = loss.get_points(
            tf.constant(featmap_sizes), tf.float32)
        num_points_per_lvl = [tf.shape(center)[0].numpy() \
                                for center in all_level_points]
        points =  tf.concat(all_level_points, axis=0)

        regress_ranges = (
            (-1.0, 64.0), 
            (64.0, 128.0), 
            (128.0, 256.0), 
            (256.0, 512.0), 
            (512.0, INF))
        regress_ranges = [tf.tile(
            tf.expand_dims(regress_ranges[i], axis=0),
            (tf.shape(all_level_points[i])[0], 1)) \
            for i in range(len(all_level_points))]
        regress_ranges = tf.concat(regress_ranges, axis=0)
        
        labels, bbox_targets, gt_ind = loss.fcos_target_single(
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