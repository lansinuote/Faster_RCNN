import torch
import torchvision
import torchvision.models.detection._utils as det_utils
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F

# RPN parameters
rpn_nms_top_n = 100
# rpn_pre_nms_top_n_train = 100
# rpn_pre_nms_top_n_test = 100
# rpn_post_nms_top_n_train = 100
# rpn_post_nms_top_n_test = 100
rpn_nms_thresh = 0.7
rpn_fg_iou_thresh = 0.7
rpn_bg_iou_thresh = 0.3
rpn_batch_size_per_image = 256
rpn_positive_fraction = 0.5
rpn_score_thresh = 0.0


def _get_top_n_idx(objectness, num_anchors_per_level):
    # type: (Tensor, List[int]) -> Tensor
    r = []
    offset = 0
    for ob in objectness.split(num_anchors_per_level, 1):
        if torchvision._is_tracing():
            1 / 0
        else:
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(rpn_nms_top_n, num_anchors)
        _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors
    return torch.cat(r, dim=1)


def filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level):
    # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop throught objectness
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = _get_top_n_idx(objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, 1e-3)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= rpn_score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, rpn_nms_thresh)

        # keep only topk scoring predictions
        keep = keep[:rpn_nms_top_n]
        boxes, scores = boxes[keep], scores[keep]

        final_boxes.append(boxes)
        final_scores.append(scores)
    return final_boxes, final_scores


proposal_matcher = det_utils.Matcher(
    rpn_fg_iou_thresh,
    rpn_bg_iou_thresh,
    allow_low_quality_matches=True,
)


def assign_targets_to_anchors(anchors, targets):
    # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
    labels = []
    matched_gt_boxes = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]

        if gt_boxes.numel() == 0:
            # Background image (negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            # used during training
            match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
            matched_idxs = proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes


fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
    rpn_batch_size_per_image, rpn_positive_fraction
)


def compute_loss(objectness, pred_bbox_deltas, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Args:
        objectness (Tensor)
        pred_bbox_deltas (Tensor)
        labels (List[Tensor])
        regression_targets (List[Tensor])

    Returns:
        objectness_loss (Tensor)
        box_loss (Tensor)
    """

    sampled_pos_inds, sampled_neg_inds = fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    box_loss = F.smooth_l1_loss(
        pred_bbox_deltas[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1 / 9,
        reduction='sum',
    ) / (sampled_inds.numel())

    objectness_loss = F.binary_cross_entropy_with_logits(
        objectness[sampled_inds], labels[sampled_inds]
    )

    return objectness_loss, box_loss
