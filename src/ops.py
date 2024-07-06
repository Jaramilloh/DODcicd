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


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
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


def calc_ratio(width, height, img_size):
    """
    Calculate the ratio of the image size.

    Args:
        width (int): The width of the image.
        height (int): The height of the image.
        img_size (int): The size of the image.
    Returns:
        ratio (float): The ratio of the image size.
        new_width (int): The new width of the image.
        new_height (int): The new height of the image.
    """
    if width > height:
        ratio = img_size / width * 1.0
        new_width = int(width * ratio)
        new_height = int(img_size)
    else:
        ratio = img_size / height * 1.0
        new_width = int(img_size)
        new_height = int(height * ratio)
    return ratio, new_width, new_height


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


def draw_bbox_w_depth_text(img, bbox, depth, thickness=2):
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


def draw_bbox_yolov8(img, bbox, cls, thickness=2):
    """
    Draw a bounding boxes on the image.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x1, y1, x2, y2) format.
        cls (float): The class.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    TEXT_COLOR = (215, 0, 0)
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    color = [
        int(255) / 255.0,
        int(0) / 255.0,
        int(255) / 255.0,
    ]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    text_to_print = str(cls)
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


def draw_bbox_w_depth(img, bbox, depth, thickness=2):
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
        img = draw_bbox_w_depth(img, bbox, depth)
    return img


def plot_dod_batch(batch, output):
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bboxes = prediction[:, :4]
        category_ids = prediction[:, 5]
        depths = prediction[:, 6]
        # bbox, category_id, depth
        for bbox, _, depth in zip(
            bboxes.detach().numpy(),
            category_ids.detach().numpy(),
            depths.detach().numpy(),
        ):
            img = draw_bbox_w_depth(img, bbox, depth)
            # img = draw_bbox_w_depth_text(img, bbox, depth)
        pred = torch.tensor(img, dtype=torch.uint8)
        pred = pred.unsqueeze(0)
        pred = pred.permute(0, 3, 1, 2)
        if i == 0:
            batchpred = pred
        else:
            batchpred = torch.cat((batchpred, pred), dim=0)
    return batchpred


def plot_dod_frame(img, output):
    """
    Visualize the prediction result for an input image.
    Args:
        img (np.ndarray): The input image array.
        nms (torch.Tensor): The predictions tensor.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    for i in range(len(output)):
        prediction = output[i].to("cpu")
        bboxes = prediction[:, :4]
        category_ids = prediction[:, 5]
        depths = prediction[:, 6]

        for bbox, _, depth in zip(
            bboxes.detach().numpy(),
            category_ids.detach().numpy(),
            depths.detach().numpy(),
        ):
            img = draw_bbox_w_depth(img, bbox, depth)
    return img


def plot_yolo_batch(batch, output, classes):
    """
    Visualize the prediction result from yolov8n for a batch.
    Args:
        batch (torch.Tensor): The input image(s) tensor.
        output (torch.Tensor): The prediction(s) tensor.
        classes (list): The list of class names.
    Returns:
        batchpred (np.ndarray): The image(s) with the bounding box and classes.
    """
    batchpred = torch.zeros(
        (batch.shape[1], batch.shape[2], batch.shape[3]), dtype=torch.float32
    )
    for i in range(len(output)):
        prediction = output[i].to("cpu")
        img = batch[i].to("cpu").permute(1, 2, 0).numpy().copy()
        img = 255 * img  # Now scale by 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bboxes = prediction[:, :4]
        category_ids = prediction[:, 5]
        # bbox, category_id,
        for (
            bbox,
            cls,
        ) in zip(bboxes.detach().numpy(), category_ids.detach().numpy()):
            img = draw_bbox_yolov8(img, bbox, classes[int(cls)])
        pred = torch.tensor(img, dtype=torch.uint8)
        pred = pred.unsqueeze(0)
        pred = pred.permute(0, 3, 1, 2)
        if i == 0:
            batchpred = pred
        else:
            batchpred = torch.cat((batchpred, pred), dim=0)
    return batchpred


def plot_yolo_frame(img, output, classes):
    """
    Visualize the prediction result from yolov8n for an image.
    Args:
        img (np.ndarray): The input image array.
        output (torch.Tensor): The prediction(s) tensor.
        classes (list): The list of class names.
    Returns:
        img (np.ndarray): The image with the bounding box and classes.
    """

    for i in range(len(output)):
        prediction = output[i].to("cpu")
        bboxes = prediction[:, :4]
        category_ids = prediction[:, 5]
        # bbox, category_id,
        for (
            bbox,
            cls,
        ) in zip(bboxes.detach().numpy(), category_ids.detach().numpy()):
            img = draw_bbox_yolov8(img, bbox, classes[int(cls)])
    return img


def draw_bbox_track_w_depth(img, bbox, depth, idtrack, thickness=2):
    """
    Draw a bounding boxes on the image representing depth value as intensity in color green with its ID.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x, y, width, height) format.
        depth (float): The depth value.
        idtrack (int): The track ID.
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

    text = f"id:{int(idtrack)}"

    scale = 0.5
    ((_, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(
        img,
        text=text,
        org=(x_min - 5, y_min - int(scale * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=green_color,
        lineType=cv2.LINE_AA,
    )
    return img


def plot_dod_frametracks(img, tracks):
    """
    Visualize the prediction result from the DOD model for a batch.
    Args:
        img (np.ndarray): The input image array.
        tracks (np.ndarray): The tracks array.
    Returns:
        batchpred (np.ndarray): The image with the bounding box, depth value and track IDs.
    """

    bboxes = tracks[:, :4]  # xyxy
    ids = tracks[:, 4]  # track id
    scores = tracks[:, 5]  # score
    cls = tracks[:, 6]  # class id
    depths = tracks[:, 7]  # depth

    for bbox, _, depth, idtrack, _ in zip(bboxes, cls, depths, ids, scores):
        img = draw_bbox_track_w_depth(img, bbox, depth, idtrack)

    return img


def draw_bbox_track_yolo(img, bbox, cls, ids, thickness=2):
    """
    Draw a bounding boxes on the image.
    Args:
        img (np.ndarray): The input image.
        bbox (tuple): The bounding box coordinates in (x1, y1, x2, y2) format.
        cls (float): The class.
        ids (int): The track ID.
        thickness (int, optional): The thickness of the bounding box. Defaults to 2.
    Returns:
        img (np.ndarray): The image with the bounding box and depth value.
    """
    TEXT_COLOR = (215, 0, 0)
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    color = [
        int(255),
        int(0),
        int(255),
    ]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    text_to_print = f"{str(cls)} id:{int(ids)}"
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


def plot_yolo_frametracks(img, tracks, classes):
    """
    Visualize the prediction result from the DOD model for a batch.
    Args:
        img (np.ndarray): The input image array.
        tracks (np.ndarray): The tracks array.
        classes (list): The list of class names.
    Returns:
        batchpred (np.ndarray): The image with the bounding box, depth value and track IDs.
    """

    bboxes = tracks[:, :4]  # xyxy
    ids = tracks[:, 4]  # track id
    scores = tracks[:, 5]  # score
    category_ids = tracks[:, 6]  # class id

    for bbox, cls, idtrack, _ in zip(bboxes, category_ids, ids, scores):
        img = draw_bbox_track_yolo(img, bbox, classes[int(cls)], idtrack)

    return img
