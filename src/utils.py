from collections import defaultdict
import numpy as np
import torch
import cv2

# import matplotlib.pyplot as plt


class MetricMonitor:
    """
    Metric Monitor class to show a loading bar along training to follow-up
    in-time proccesed batch and epoch, with its corresponding metrics.
    """

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, updated=False):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        if updated == False:
            metric["avg"] = metric["val"] / metric["count"]
        else:
            metric["avg"] = val

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def ltwh2xyxy(x):
    """
    It converts the bounding box from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (left, top, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] + x[:, 0]  # width
    y[:, 3] = x[:, 3] + x[:, 1]  # height
    return y


def xyxy2ltwh(x):
    """
    Convert nx4 bounding boxes from [x1, y1, x2, y2] to [x1, y1, w, h], where xy1=top-left, xy2=bottom-right

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, x, y) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (left, right, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def map_variable_to_green(variable):
    """
    Map depth value into green intensity color
    Args:
        variable (float): The input depth value.
    Returns:
        green_bgr (np.ndarray): The green intensity color in BGR format.
    """
    min_value = 15
    max_value = 25
    variable = max(min_value, min(variable, max_value))
    normalized_variable = (variable - min_value) / (max_value - min_value)
    hue = 60
    saturation = int(255)
    value = 255 - int(255 * (1 - normalized_variable))
    green_bgr = np.array([[[hue, saturation, value]]], dtype=np.uint8)
    green_bgr = cv2.cvtColor(green_bgr, cv2.COLOR_HSV2BGR)
    return green_bgr[0, 0]


def visualize_depth_bbox_text(img, bbox, depth, thickness=2):
    """
    Draw a bounding boxes on the image displaying the depth value representation.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x, y, width, height) format.
        depth (float): The depth value.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    TEXT_COLOR = (215, 0, 0)
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    green_color = map_variable_to_green(depth)
    green_color = [green_color[0] / 1.0, green_color[1] / 1.0, green_color[2] / 1.0]
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=green_color, thickness=thickness
    )
    text_to_print = str(np.round(depth, 2))
    # ((text_width, text_height), _)
    ((_, text_height), _) = cv2.getTextSize(
        text_to_print, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
    )
    cv2.putText(
        img,
        text=text_to_print,
        org=(x_min - 5, y_min - int(0.32 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.32,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_depth_bbox_yolo_text(img, bbox, depth, thickness=2):
    """
    Draw a bounding boxes on the image displaying the depth value representation.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x1, y1, x2, y2) format.
        depth (float): The depth value.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    TEXT_COLOR = (215, 0, 0)
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    green_color = map_variable_to_green(depth)
    green_color = [
        int(green_color[0]) / 255.0,
        int(green_color[1]) / 255.0,
        int(green_color[2]) / 255.0,
    ]
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=green_color, thickness=thickness
    )
    text_to_print = str(np.round(depth, 2))
    # ((text_width, text_height), _)
    ((_, text_height), _) = cv2.getTextSize(
        text_to_print, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1
    )
    cv2.putText(
        img,
        text=text_to_print,
        org=(x_min - 5, y_min - int(0.32 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.32,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_depth_bbox(img, bbox, depth, thickness=2):
    """
    Draw a bounding boxes on the image representing depth value as intensity in color green.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x, y, width, height) format.
        depth (float): The depth value.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    green_color = map_variable_to_green(depth)
    green_color = [green_color[0], green_color[1], green_color[2]]
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=green_color, thickness=thickness
    )
    # print(f"depth {depth[0]:.2f} color {green_color}")
    return img


def visualize_depth_bbox_yolo(img, bbox, depth, thickness=2):
    """
    Draw a bounding boxes on the image representing depth value as intensity in color green.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x, y, width, height) format.
        depth (float): The depth value.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    green_color = map_variable_to_green(depth)
    green_color = [int(green_color[0]), int(green_color[1]), int(green_color[2])]
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), color=green_color, thickness=thickness
    )
    # print(f"depth {depth[0]:.2f} color {green_color}")
    return img


# img, nms, category_id_to_name, name
def visualize_pred(img, nms):
    """
    Visualize the prediction result from the model for a single image.
    Args:
        img (torch.Tensor): The input image(s) tensor.
        nms (torch.Tensor): The prediction(s) tensor.
    Returns:
        img (np.ndarray): The image(s) with the bounding box and depth value.

    """
    prediction = nms[0].to("cpu")
    img = img[0].to("cpu").permute(1, 2, 0).numpy().copy()
    img = 255 * img  # Now scale by 255
    img = img.astype(np.uint8)
    bboxes = prediction[:, :4]
    category_ids = prediction[:, 5]
    depths = prediction[:, 6]
    # bbox, category_id, depth
    for bbox, _, depth in zip(
        bboxes.detach().numpy(), category_ids.detach().numpy(), depths.detach().numpy()
    ):
        img = visualize_depth_bbox_yolo(img, bbox, depth)
    return img


# img, nms, category_id_to_name, name
def visualize_batch(batch, output):
    """
    Visualize the prediction result from the model for a batch.
    Args:
        batch (torch.Tensor): The input image(s) tensor.
        nms (torch.Tensor): The prediction(s) tensor.
    Returns:
        batchpred (np.ndarray): The image(s) with the bounding box and depth value.
    """
    batchpred = torch.zeros(
        (batch.shape[1], batch.shape[2], batch.shape[3]), dtype=torch.float32
    )
    for i in range(len(output)):
        prediction = output[i].to("cpu")
        img = batch[i].to("cpu").permute(1, 2, 0).numpy().copy()
        img = 255 * img  # Now scale by 255
        img = img.astype(np.uint8)
        bboxes = prediction[:, :4]
        category_ids = prediction[:, 5]
        depths = prediction[:, 6]
        # bbox, category_id, depth
        for bbox, _, depth in zip(
            bboxes.detach().numpy(),
            category_ids.detach().numpy(),
            depths.detach().numpy(),
        ):
            img = visualize_depth_bbox_yolo(img, bbox, depth)
            # img = visualize_depth_bbox_yolo_text(img, bbox, depth)
        pred = torch.tensor(img, dtype=torch.uint8)
        pred = pred.unsqueeze(0)
        pred = pred.permute(0, 3, 1, 2)
        if i == 0:
            batchpred = pred
        else:
            batchpred = torch.cat((batchpred, pred), dim=0)
    return batchpred
