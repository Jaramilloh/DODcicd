import sys
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x1, y1, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0]  # top left x
    y[..., 1] = x[..., 1]  # top left y
    y[..., 2] = x[..., 0] + x[..., 2]  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3]  # bottom right y
    return y


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)  # lt and rb is in range[0, self.reg_max-1]
    # width and height of bounding box are in range [0, 2*(self.reg_max-1)] owing to (x2y2-x1y1=rb+lt)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
        bs, n_boxes, n_anchors, -1
    )
    return bbox_deltas.amin(3).gt_(eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(
            b2_x1
        )  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (
                    torch.atan(w2 / h2) - torch.atan(w1 / h1)
                ).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return (
            iou - (c_area - union) / c_area
        )  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w) -> mask with topk predictions for each gt
        overlaps (Tensor): shape(b, n_max_boxes, h*w) -> CIoU result for predictions with gt
    Return:
        target_gt_idx (Tensor): shape(b, h*w) -> tensor with corresponding gt bbox index for each predicted anchor point
        fg_mask (Tensor): shape(b, h*w) -> mask with final (positive) anchor points within the predicted batch
        mask_pos (Tensor): shape(b, n_max_boxes, h*w) -> mask with topk predictions for each gt (without one anchor assigned to multiple gt bboxes)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(
            -1, n_max_boxes, -1
        )  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        is_max_overlaps = torch.zeros(
            mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
        )
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(
            mask_multi_gts, is_max_overlaps, mask_pos
        ).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos  # (b, h*w), (b, h*w), (b, n_max_boes, h*w)


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
    which combines both classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(
        self, device="cpu", topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9
    ):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device
        self.bs = None
        self.n_max_boxes = None

    @torch.no_grad()
    def forward(
        self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, gt_depths, mask_gt
    ):
        """
        Compute the task-aligned assignment.
        Reference https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_depths (Tensor): shape(bs, n_max_boxes, 1)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
            target_gt_idx (Tensor): shape(bs, num_total_anchors)
            target_depths (Tensor): shape(bs, num_total_anchors)
        """
        device = self.device
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )

        # Assigned target
        target_labels, target_bboxes, target_scores, target_depths = self.get_targets(
            gt_labels, gt_bboxes, gt_depths, target_gt_idx, fg_mask
        )

        # Normalize
        align_metric *= mask_pos  # (b, n_max_boes, h*w)
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # (b, max_num_obj)
        pos_overlaps = (overlaps * mask_pos).amax(
            axis=-1, keepdim=True
        )  # (b, max_num_obj)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )

        target_scores = (
            target_scores * norm_align_metric
        )  # (b, h*w, nc) * (b, h*w, 1) = (b, h*w, nc)

        return (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask.bool(),
            target_gt_idx,
            target_depths,
        )  # (b, h*w), (b, h*w, 4), (b, h*w, nc), (b, h*w), (b, h*w), (b, h*w)

    def get_pos_mask(
        self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
    ):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        # Get mask with positive anchor centers relative to groud truth boxes
        mask_in_gts = select_candidates_in_gts(
            anc_points, gt_bboxes
        )  # (b, max_num_obj, h*w)
        # Calculate align metric (CIoU * pred score) between gt masked boxes and predicted boxes
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )  # (b, max_num_obj, h*w) align metric, (b, max_num_obj, h*w) CIoU
        # Get mask with topk predicted boxes for each gt masked box
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )  # (b, max_num_obj, h*w)
        # Merge all mask to a final mask with top k predicted boxes with positive anchor centers relative to masked gt boxes
        mask_pos = (
            mask_topk * mask_in_gts * mask_gt
        )  # (b, max_num_obj, h*w) * (b, max_num_obj, h*w) * (b, max_num_obj, 1) = (b, max_num_obj, h*w)
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )

        ind = torch.zeros(
            [2, self.bs, self.n_max_boxes], dtype=torch.long
        )  # 2, b, max_num_obj
        ind[0] = (
            torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        )  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the predicted scores for all anchor points for each masked ground truth class
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][
            mask_gt
        ]  # (b, max_num_obj, h*w)

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[
            mask_gt
        ]  # (b, max_num_obj, 1, 4)
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[
            mask_gt
        ]  # (b, 1, h*w, 4)
        # Compare each gt boxes with all anchor points predicted boxes
        overlaps[mask_gt] = (
            bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
        )  # (b, max_num_obj, h*w)

        # Calculate the align metric for each: CIoU * Pred cls score
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """

        # Find the top k larger align predicted boxes for each ground truth
        topk_metrics, topk_idxs = torch.topk(
            metrics, self.topk, dim=-1, largest=largest
        )  # (b, max_num_obj, topk)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(
                topk_idxs
            )
        if topk_mask is not None:
            topk_idxs.masked_fill_(~topk_mask, 0)  # (b, max_num_obj, topk)
        else:
            # Handle the case where topk_mask is None
            print("topk_mask is NoneType object")
            sys.exit(1)

        # Get the a mask with the top k targets
        count_tensor = torch.zeros(
            metrics.shape, dtype=torch.int8, device=topk_idxs.device
        )
        ones = torch.ones_like(
            topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device
        )
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)  # (b, max_num_obj, h*w)

    def get_targets(self, gt_labels, gt_bboxes, gt_depths, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            gt_depths (Tensor): Ground truth depth for each bounding box, shape (b, max_num_obj, 1).
            target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                - target_labels (Tensor): Shape (b, h*w), containing the target labels for
                                          positive anchor points.
                - target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                          for positive anchor points.
                - target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                          for positive anchor points, where num_classes is the number
                                          of object classes.
        """

        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Assigned target depths, (b, max_num_obj, 4) -> (b, h*w)
        target_depths = gt_depths.view(-1)[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)
        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, nc)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(
            1, 1, self.num_classes
        )  # (b, h*w, nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return (
            target_labels,
            target_bboxes,
            target_scores,
            target_depths,
        )  # (b, h*w), (b, h*w, 4), (b, h*w, nc), (b, h*w, 1)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
        0, reg_max - 0.01
    )  # dist (lt, rb)


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """
        Compute IoU loss.

        Args:
            pred_dist (Tensor): Shape(b, h*w, 4*reg_max), containing the predicted bounding
                                                      boxes distribution.
            pred_bboxes (Tensor): Shape(b, h*w, 4), containing the predicted bounding boxees
            anchor_points (Tensor): Shape(h*w, 2), containing the anchor points of the grids
            target_bboxes (Tensor): Shape (b, h*w, 4), containing the target bounding boxes
                                                      for positive anchor points.
            target_scores (Tensor): Shape (b, h*w, num_classes), containing the target scores
                                                      for positive anchor points, where num_classes is the number
                                                      of object classes.
            target_scores_sum (Tensor): Shape (1), containing the sum over all target scores.
            fg_mask (Tensor): Shape (b, h*w), indicating the positive
                              (foreground) anchor points.

        Returns:
            (Tuple[Tensor, Tensor]): A tuple containing the following tensors:
                - loss_iou (Tensor) CIoU regression loss
                - loss_dfl (Tensor) DFL regression loss
        """

        # IoU loss
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = (
                self._df_loss(
                    pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Args:
            pred_dist (Tensor): Shape(b, h*w, 4*reg_max), containing the predicted bounding
                                        boxes distribution.
            target (Tensor): Shape(h*w[fg_mask], 4), containing the target dist(ltrb)

        Returns:
            loss_dfl (Tensor): Shape [1] Distribution Focal Loss (DFL) proposed
                                    in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left (h*w[fg_mask], 4)
        tr = tl + 1  # target right (h*w[fg_mask], 4)
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class v8DetectionLoss:
    def __init__(
        self, stride, device, nclasses=80, reg_max=4
    ):  # model must be de-paralleled

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="mean")

        self.stride = stride  # stride for each head at model output
        self.nc = nclasses  # number of classes
        self.no = reg_max * 4 + nclasses + 1
        self.reg_max = reg_max
        self.device = device

        self.use_dfl = reg_max > 1

        self.assigner = TaskAlignedAssigner(
            device=self.device, topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)

        self.box = 7.5  # 3.5#7.5
        self.cls = 0.5  # 0.1#0.5
        self.dfl = 1.5  # 0.5#1.5
        self.dpt = 0.05

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            # Reduce dist (ltrb*reg_max) predictions into a box xyxy prediction
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )  # softmax((b, h*w, 4, 4), dim=-1) * (4) -> (b, h*w, 4)
        return dist2bbox(pred_dist, anchor_points, xywh=False).clamp_(
            0.0
        )  # (b, h*w, 4)

    def __call__(self, feats, batch, depth=True):

        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, depth

        if depth:
            # Extract predictions from each head at different strides
            pred_distri, pred_scores, pred_depth = torch.cat(
                [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
            ).split((self.reg_max * 4, self.nc, 1), 1)

            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (b, h*w, nc)
            pred_distri = pred_distri.permute(
                0, 2, 1
            ).contiguous()  # (b, h*w, 4*reg_max)
            pred_depth = pred_depth.permute(0, 2, 1).contiguous()  # (b, h*w, 1)
        else:
            no = self.no - 1
            pred_distri, pred_scores = torch.cat(
                [xi.view(feats[0].shape[0], no, -1) for xi in feats], 2
            ).split((self.reg_max * 4, self.nc), 1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (b, h*w, nc)
            pred_distri = pred_distri.permute(
                0, 2, 1
            ).contiguous()  # (b, h*w, 4*reg_max)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # imgsz = (
        #    torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
        #    * self.stride[0]
        # )  # image size (h,w)

        # Get anchor point centers from output grids and its corresponding stride
        anchor_points, stride_tensor = make_anchors(
            feats, self.stride, 0.5
        )  # (h*w, 2), (h*w, 1)

        # Extract ground truth boxes and labels
        gt_bboxes = xywh2xyxy(batch["bboxes"].to(self.device))  # (b, max_num_obj, 4)
        gt_labels = batch["cls"].to(self.device)  # (b, max_num_obj, 1)
        gt_depths = batch["depths"].to(self.device)  # (b, max_num_obj, 1)

        # Get gt mask to filter batch padding
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # (b, max_num_obj, 1)

        # Get pboxes relative to anchor points
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Apply TOOD's TaskAssigner to assign the best predictions to each ground truth
        (
            _,
            target_bboxes,
            target_scores,
            fg_mask,
            _,  # target_gtidx,
            target_depths,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_depths,
            mask_gt,
        )  # _, (b, h*w, 4), (b, h*w, nc), (b, h*w), _

        target_scores_sum = max(target_scores.sum(), 1)

        # classification loss
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        # depth loss
        if depth:
            pred_depth = pred_depth.view(batch_size, -1)
            loss[3] = self.mse(pred_depth[fg_mask], target_depths[fg_mask])  # MSE

        loss[0] *= self.box  # box gain
        loss[1] *= self.cls  # cls gain
        loss[2] *= self.dfl  # dfl gain
        loss[3] *= self.dpt  # depth gain

        return (
            loss[3] * batch_size,
            loss[0:3].sum() * batch_size,
            loss.detach(),
        )  # dpt_loss, sum(obj det losses), sum(all losses), losses_detached
