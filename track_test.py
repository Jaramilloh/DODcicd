#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from copy import deepcopy
import yaml
import argparse

import numpy as np
import cv2
import torch

from src.inference import non_max_suppression

from src.ops import (
    calc_ratio,
    plot_dod_frame,
    plot_yolo_frame,
    plot_dod_frametracks,
    plot_yolo_frametracks,
)

from src.model import LoadModel

from src.utils import LOGGER
from src.utils import check_yaml, yaml_load
from src.utils import IterableSimpleNamespace

from trackers.bot_sort import BOTSORT
from trackers.byte_tracker import BYTETracker

LOGGER.info(" ")
LOGGER.info("Starting the program")

TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


# ------------- Object Tracking Functions


def on_predict_start(
    trackers, persist=False, tracker="trackers/cfg/bytetrack.yaml", mode="stream", bs=1
):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        trackers (list): Intiaziled trackers.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
        tracker (str, optional): The tracker configuration file. Defaults to 'trackers/cfg/bytetrack.yaml'.
        mode (str): The mode of the tracker.
        bs (int): The batch size.
    Returns:
        trackers (list): The initialized trackers.
    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    if trackers and persist:
        return

    tracker = check_yaml(tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    # print(cfg)

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(
            f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'"
        )

    trackers = []
    for _ in range(bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if mode != "stream":  # only need one tracker for other modes.
            break
    return trackers


def on_predict_postprocess_end(
    trackers=None, prediction=None, im0=None, version="v2", persist=False
):
    """
    Postprocess detected boxes and update with object tracking.

    Args:
        trackers (list): The tracker object containing the predictions.
        prediction (torch.Tensor): The prediction tensor.
        im0 (np.ndarray): The image np array.
        version (str): The version of the model.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    Returns:
        objtracks (np.ndarray): The tracked objects.
    """

    tracker = trackers[0]

    if not persist:
        tracker.reset()

    det = (
        prediction[0].cpu().numpy()
    )  # <class 'numpy.ndarray'> with shape(n, 7) bbox xyxy, score, cls, nmask

    if len(det) == 0:
        return

    objtracks = tracker.update(det, im0)

    if len(objtracks) == 0:
        return

    idx = objtracks[:, -1].astype(int)

    if version != "yolo":
        depth = det[:, -1]
        idx = objtracks[:, -1].astype(int)
        objtracks = np.column_stack((objtracks[:, :-1], depth[idx]))
    else:
        objtracks = np.array((objtracks[:, :-1]))
    return objtracks


# -------------


class Parameters:
    """
    Define the parameters for the video processing.
    """

    def __init__(self):
        self.cwd = os.getcwd()
        self.root = os.path.dirname(self.cwd)
        self.classes = {0: "fruit"}
        self.params = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "img_size": 320,
            "reg_max": 4,
            "batch_size": 32,
            "classes": self.classes,
            "cwd": self.cwd,
            "root": self.root,
        }


params = Parameters()

# --------- MAIN


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Track Test Script")
    parser.add_argument(
        "--version", type=str, default="v2", help="Version of the model: v1, v2 or yolo"
    )
    args = parser.parse_args()

    classes = params.classes
    device = params.params["device"]
    ver = args.version

    if ver == "yolo":
        with open(f"{params.root}/DODcicd/models/coco.yaml", encoding="utf-8") as f:
            classes = yaml.safe_load(f)["names"]
            params.classes = classes

    model, inference = LoadModel(version=ver, improved=False, params=params)

    cap = cv2.VideoCapture("videos/oranges1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)

    startshot = False
    img_size = params.params["img_size"]

    trckrs = []
    trckrs = on_predict_start(
        trckrs, persist=False, tracker="trackers/cfg/bytetrack.yaml", mode="none"
    )

    while True:

        ret, frame = cap.read()

        if ret:

            # --- Frame preprocessing

            if not startshot:

                width, height, _ = frame.shape
                _, new_width, new_height = calc_ratio(width, height, img_size)
                startshot = True

            frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = torch.tensor(deepcopy(frame), dtype=torch.float32) / 255.0
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)

            # --- Object Detection

            if ver == "yolo":
                _, x = model(img.to(device))
            else:
                x = model(img.to(device))

            for predheads in x:
                if torch.sum(predheads) == 0:
                    print("NaN in prediction")
                    break

            y = inference(x)

            output = non_max_suppression(
                y,
                conf_thres=0.2,
                iou_thres=0.1,
                max_det=300,
                nc=len(classes),
                multi_label=False,
            )  # bbox xyxy, score, cls, nmask

            # --- Object Tracking

            # print(output)
            tracks = None
            tracks = on_predict_postprocess_end(
                trackers=trckrs,
                prediction=output,
                im0=frame,
                version=ver,
                persist=True,
            )

            # --- Visualization predictions

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            framedraw = deepcopy(frame)

            if ver == "yolo":
                objresult = plot_yolo_frame(framedraw, output, classes)
            else:
                objresult = plot_dod_frame(framedraw, output)

            cv2.imshow("ObjDet", objresult)

            # --- Visualization tracking

            if tracks is not None:

                framedraw = deepcopy(frame)

                if ver == "yolo":
                    trckresult = plot_yolo_frametracks(framedraw, tracks, classes)
                else:
                    trckresult = plot_dod_frametracks(framedraw, tracks)
                cv2.imshow("object tracking", trckresult)

            # --------------------

            cv2.imshow("press `q` to exit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    print("cap closed")
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    LOGGER.info("Finishing the program successfully")
    sys.exit()

# --------
