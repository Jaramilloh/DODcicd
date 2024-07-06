import time
import torch
from torch import nn
import torchvision
from src.ops import xywh2xyxy, box_iou


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16, device="cpu"):
        """Initialize a convolutional layer with a given number of input channels."""
        super(DFL, self).__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False).to(device)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        result = self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )
        return result


class Inference(nn.Module):
    def __init__(self, nclasses=1, stride=None, reg_max=1, device="cpu", tempscale=1.0):
        """
        Inference module of DOD.
        Args:
            nclasses (int): number of classes in the dataset.
            stride (list[int]): stride of each head in the model.
            reg_max (int): maximum value of regression.
            device (str): device to run the model.
        """
        super(Inference, self).__init__()
        self.stride = stride
        self.nc = nclasses
        self.reg_max = reg_max
        self.no = self.reg_max * 4 + nclasses + 1
        self.dfl = DFL(
            self.reg_max, device=device
        )  # if self.reg_max > 1 else nn.Identity()
        self.tempscale = tempscale

    def forward(self, feats):
        """
        Extract predictions from each head at different strides.
        Args:
            feats (list[torch.Tensor]): list of predictions from each head.
        Returns:
            torch.Tensor: concatenated postprocessed predictions from all heads.
        """
        # Extract predictions from each head at different strides
        pred_distri, pred_scores, pred_depth = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc, 1), 1)
        pred_scores = pred_scores.permute(0, 1, 2).contiguous()  # (b, nc, h*w)
        pred_distri = pred_distri.permute(0, 1, 2).contiguous()  # (b, 4*reg_max, h*w)
        pred_depth = pred_depth.permute(0, 1, 2).contiguous()  # (b, 1, h*w)
        # print(pred_depth)
        # Get anchor point centers from output grids and its corresponding stride
        anchors, strides = (
            x.transpose(0, 1) for x in self.make_anchors(feats, self.stride, 0.5)
        )
        # Decode reg_max*4 prediction to cxywh bounding box prediction
        dbox = (
            self.dist2bbox(
                self.dfl(pred_distri), anchors.unsqueeze(0), xywh=True, dim=1
            ).clamp_(0.0)
            * strides
        )
        pred_scores = pred_scores / self.tempscale
        y = torch.cat(
            (dbox, pred_scores.sigmoid(), pred_depth), 1
        )  # (bs, 4 + nclasses + depth, h*w)
        return y

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy).
        width and height of bounding box are in range [0, 2*(self.reg_max-1)] owing to (x2y2-x1y1=rb+lt)
        """
        lt, rb = distance.chunk(2, dim)  # lt and rb is in range[0, self.reg_max-1]
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
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
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)


class InferenceYolo(nn.Module):
    """
    Inference module of DOD.
    Args:
        nclasses (int): number of classes in the dataset.
        stride (list[int]): stride of each head in the model.
        reg_max (int): maximum value of regression.
        device (str): device to run the model.
    """

    def __init__(self, nclasses=1, stride=None, reg_max=16, device="cpu"):
        super(InferenceYolo, self).__init__()
        self.stride = stride
        self.nc = nclasses
        self.reg_max = reg_max
        self.no = self.reg_max * 4 + nclasses
        self.dfl = DFL(self.reg_max).to(
            device
        )  # if self.reg_max > 1 else nn.Identity()

    def forward(self, feats):
        # Extract predictions from each head at different strides
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 1, 2).contiguous()
        pred_distri = pred_distri.permute(0, 1, 2).contiguous()
        # Get anchor point centers from output grids and its corresponding stride
        anchors, strides = (
            x.transpose(0, 1) for x in self.make_anchors(feats, self.stride, 0.5)
        )
        # Decode reg_max*4 prediction to cxywh bounding box prediction
        dbox = (
            self.dist2bbox(
                self.dfl(pred_distri), anchors.unsqueeze(0), xywh=True, dim=1
            ).clamp_(0.0)
            * strides
        )
        y = torch.cat((dbox, pred_scores.sigmoid()), 1)  # (bs, 4 + nclasses, h*w)
        return y

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy).
        width and height of bounding box are in range [0, 2*(self.reg_max-1)] owing to (x2y2-x1y1=rb+lt)
        """
        lt, rb = distance.chunk(2, dim)  # lt and rb is in range[0, self.reg_max-1]
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
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
            stride_tensor.append(
                torch.full((h * w, 1), stride, dtype=dtype, device=device)
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)


def non_max_suppression(
    pred,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    prediction = pred  # (bs, 4 + nclasses + depth scalar, h*w) ; h*w = num_boxes or predicted boxes

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates (bs, h*w)

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [
        torch.zeros((0, 6 + nm), device=prediction.device)
    ] * bs  # list with lenght=bs of (0, 4;bbox + 1;score + 1;cls + nm)
    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply constraints
        x = x.transpose(0, -1)[xc[xi]]  # confidence (num_candidates, 4 + nclasses + nm)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (x1y1x2y2, conf, cls)
        box, cls, mask = x.split(
            (4, nc, nm), 1
        )  # (num_candidates, 4), (num_candidates, n_classes), (num_candidates, nm)
        box = xywh2xyxy(box).clamp_(
            0.0
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > conf_thres
            ]  # (num_candidates, 4;bbox + 1;score + 1;cls + nm)

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        # Sort by confidence and remove excess boxes
        x = x[
            x[:, 4].argsort(descending=True)[:max_nms]
        ]  # (num_candidates, 4;bbox + 1;score + 1;cls + nm)

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes (num_candidates, 1;cls)

        # boxes (offset by class), scores
        boxes, scores = (
            x[:, :4] + c,
            x[:, 4],
        )  # (num_candidates, 4), (num_candidates, 1)

        # NMS
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # (num_survivor_bboxes)
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            # break  # time limit exceeded

    return output
