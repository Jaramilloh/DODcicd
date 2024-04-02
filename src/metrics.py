import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
import logging

# Create a custom logger
LOGGER = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file_metrics.log")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
LOGGER.addHandler(c_handler)
LOGGER.addHandler(f_handler)


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def plt_settings(rcparams=None, backend="Agg"):
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Usage:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """

    if rcparams is None:
        rcparams = {"font.size": 11}

    def decorator(func):
        """Decorator to apply temporary rc parameters and backend to a function."""

        def wrapper(*args, **kwargs):
            """Sets rc parameters and backend, calls the original function, and restores the settings."""
            original_backend = plt.get_backend()
            plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


@plt_settings()
def plot_mc_curve(
    px,
    py,
    save_dir=Path("mc_curve.png"),
    names=(),
    xlabel="Confidence",
    ylabel="Metric",
    on_plot=None,
):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(
        px,
        y,
        linewidth=3,
        color="blue",
        label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(
                px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}"
            )  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(
        px,
        py.mean(1),
        linewidth=3,
        color="blue",
        label="all classes %.3f mAP@0.5" % ap[:, 0].mean(),
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Arguments:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp,
    conf,
    pred_cls,
    target_cls,
    plot=False,
    on_plot=None,
    save_dir=Path(),
    names=(),
    eps=1e-16,
    prefix="",
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = np.interp(
            -px, -conf[i], recall[:, 0], left=0
        )  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [
        v for k, v in names.items() if k in unique_classes
    ]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(
            px, py, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot
        )
        plot_mc_curve(
            px,
            f1,
            save_dir / f"{prefix}F1_curve.png",
            names,
            ylabel="F1",
            on_plot=on_plot,
        )
        plot_mc_curve(
            px,
            p,
            save_dir / f"{prefix}P_curve.png",
            names,
            ylabel="Precision",
            on_plot=on_plot,
        )
        plot_mc_curve(
            px,
            r,
            save_dir / f"{prefix}R_curve.png",
            names,
            ylabel="Recall",
            on_plot=on_plot,
        )

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
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


def ltwh2xyxy(x):
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


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.array): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = (
            np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        )
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, labels):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class).
            labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
                                  Each row should contain (class, x1, y1, x2, y2).
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def get_matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (
            (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)
        )  # remove background class if task=detect

    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn as sn

        array = self.matrix / (
            (self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
        )  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.9)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore"
            )  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            g = sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            )
            g.set_facecolor((1, 1, 1))
            g.set_xticklabels(g.get_xmajorticklabels(), fontsize=8)
            g.set_yticklabels(g.get_ymajorticklabels(), fontsize=8)
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """
        Print the confusion matrix to the console.
        """
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))


class v8DetectionValidator:

    """
    Validation class that calculates the average precision metric over different
    iou thresholds given a set of predicted bounding boxes and its corresponding targetrs.

    Args:
        device (String): device to proccess the tensors
        nclasses (int): number of classes
        classes (list): name of the classes
        plot (bool): if plotting the results
    """

    def __init__(self, device, nclasses=80, classes=()):
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.nc = nclasses  # number of classes
        self.device = device
        self.save_dir = Path(os.getcwd() + "/results")
        self.classes = classes
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.on_plot = None

    def __call__(self, pred, batch, depth=False, plots=False):

        stats = []
        nc = self.nc

        # tp, fp, p, r, f1, mp, mr, map50, ap50, map
        _, _, p, r, _, mp, mr, map50, ap50, mAP = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        # Extract ground truth boxes and labels
        gt_bboxes = ltwh2xyxy(batch["bboxes"].to(self.device))  # (b, max_num_obj, 4)
        gt_labels = batch["cls"].to(self.device)  # (b, max_num_obj, 1)

        # Get gt mask to filter batch padding
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # (b, max_num_obj, 1)

        seen = 0
        # Get correct anchor matrix for eatch image at batch
        for bs in range(mask_gt.shape[0]):
            seen += 1
            gt_bs_bboxes = gt_bboxes[bs][mask_gt[bs].bool().squeeze(-1)]  # (gt_obj, 4)
            gt_bs_labels = gt_labels[bs][mask_gt[bs].bool().squeeze(-1)]  # (gt_obj, 1)
            label_bs = torch.cat((gt_bs_labels, gt_bs_bboxes), dim=-1)  # (gt_obj, 5)
            if depth == False:
                detection_bs = pred[bs]  # (num_pred, 4;bbox+1;score+1;cls)
            else:
                detection_bs = pred[bs][:, :6]  # (num_pred, 4;bbox+1;score+1;cls)
            correct_bboxes = self._process_batch(
                detection_bs, label_bs
            )  # (num_pred, len(iouv))
            self.confusion_matrix.process_batch(detection_bs, label_bs)
            stats.append(
                (
                    correct_bboxes,
                    detection_bs[:, 4],
                    detection_bs[:, 5],
                    gt_bs_labels.squeeze(-1),
                )
            )  # (correct, conf, pcls, tcls)

        # Compute metrics
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            #  tp, fp, p, r, f1, ap, ap_class
            _, _, p, r, _, ap, _ = ap_per_class(
                *stats, plot=plots, save_dir=self.save_dir, names=self.classes
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(
            stats[3].astype(int), minlength=nc
        )  # number of targets per class

        result = {
            "seen": seen,
            "instances": nt.sum(),
            "P": mp,
            "R": mr,
            "mAP50": map50,
            "mAP50-95": mAP,
        }

        if plots == True:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir,
                    names=self.classes.values(),
                    normalize=normalize,
                    on_plot=self.on_plot,
                )

        return result

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]
        for i in range(len(self.iouv)):
            x = torch.where(
                (iou >= self.iouv[i]) & correct_class
            )  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = (
                    torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                    .cpu()
                    .numpy()
                )  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=detections.device)
