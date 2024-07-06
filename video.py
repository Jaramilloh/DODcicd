#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging

from copy import deepcopy

import queue
from threading import Thread
from threading import Event

from time import sleep

import numpy as np
import datetime as dt
import cv2

import torch

from src.inference import Inference, non_max_suppression
from src.ops import plot_dod_batch

from src.model import ConvModule  # pylint: disable=unused-import
from src.model import Bottleneck  # pylint: disable=unused-import
from src.model import C2f  # pylint: disable=unused-import
from src.model import SPPF  # pylint: disable=unused-import
from src.model import DetectionHead  # pylint: disable=unused-import
from src.model import DODv2

from pytorch_model_summary import summary

# --------- Define Logging handler
LOGGER = logging.getLogger(__name__)
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("video-file.log")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)
LOGGER.addHandler(c_handler)
LOGGER.addHandler(f_handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Handle the uncaught exception and log it.
    Args:
        exc_type (type): The exception type.
        exc_value (Exception): The exception value.
        exc_traceback (traceback): The traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    LOGGER.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
# -------------
# --------- GLOBALS

stop_threads = Event()


def clear_q(q_frame, interval=10):
    """
    Clear the queue to avoid memory overload.

    Args:
        q_frame (queue): The queue to clear.
        interval (int): The interval to clear the queue.
    """
    while not stop_threads.is_set():
        sleep(interval)
        with q_frame.mutex:  # Thread-safe operation
            q_frame.queue.clear()
            print("Queue cleared to avoid memory overload.")


def clear_b(q_batch, interval=10):
    """
    Clear the batch queue to avoid memory overload.

    Args:
        q_batch (queue): The batch queue to clear.
        interval (int): The interval to clear the batch queue.
    """
    while not stop_threads.is_set():
        sleep(interval)
        with q_batch.mutex:  # Thread-safe operation
            q_batch.queue.clear()
            print("Batch Queue cleared to avoid memory overload.")


def clear_pred(q_out, interval=10):
    """
    Clear the prediction queue to avoid memory overload.

    Args:
        q_out (queue): The prediction queue to clear.
        interval (int): The interval to clear the prediction queue.
    """
    while not stop_threads.is_set():
        sleep(interval)
        with q_out.mutex:  # Thread-safe operation
            q_out.queue.clear()
            print("Pred Queue cleared to avoid memory overload.")


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
q = queue.Queue()
b = queue.Queue()
pred = queue.Queue()

# ---------
# --------- Functions to acquire VIDEO


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


def ReadFrame(q_frame=None):
    """
    Thread to read the frame from the video.

    Args:
        q_frame (queue): The frame queue to read.
    """
    cap = cv2.VideoCapture("videos/oranges1.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    print("cap set done")
    startshot = False
    img_size = params.params["img_size"]
    while not stop_threads.is_set():
        ret, frame = cap.read()
        if not ret:
            q_frame.put(None)
            stop_threads.set()
            cap.release()
            break
        if not startshot:
            width, height, _ = frame.shape
            _, new_width, new_height = calc_ratio(width, height, img_size)
            print(f"size of frames from {width} {height} to  {new_width} {new_height}")
            startshot = True
        resized = cv2.resize(frame, (new_width, new_height))
        frame = deepcopy(resized)
        q_frame.put(resized)
        cv2.imshow("incoming video stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_threads.set()
            break
    print("cap released")
    q_frame.put(None)
    print("cap closed")


def BatchConstruct(q_frame=None, q_batch=None):
    """
    Thread to construct the batch from the frame.

    Args:
        q_frame (queue): The frame queue to construct the batch.
        q_batch (queue): The batch queue to construct.
    """
    batch_size = params.params["batch_size"]
    while not stop_threads.is_set():
        item = q_frame.get()
        q_frame.task_done()
        frame = deepcopy(item)
        if frame is None:  # Check for sentinel value and exit if found
            q_batch.put(None)
            break
        batch = []
        i = 0
        while i < batch_size and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch.append(frame)
            item = q_frame.get()
            q_frame.task_done()
            frame = deepcopy(item)
            i += 1
        imgs = np.stack(batch, axis=0)
        # print(f'imgs shape {imgs.shape}')
        imgs = torch.tensor(imgs, dtype=torch.float32) / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        # print(f'imgs tensor shape {imgs.shape}')
        q_batch.put(imgs)
    q_batch.put(None)


# ----------

# --------- Functions for Object Detection MODEL
def InferencePass(
    q_batch=None, q_out=None, obj_model=None, postprocess=None, cls=None, dev=None
):
    """
    Thread to perform inference pass for object detection.

    Args:
        q_batch (queue): The batch queue for inference.
        q_out (queue): The output queue for inference.
        obj_model (DODv2): The object detection model.
        postprocess (Inference): The postprocess for inference.
        cls (dict): The classes for object detection.
        dev (torch.device): The device for object detection.
    """
    while not stop_threads.is_set():
        item = q_batch.get()
        q_batch.task_done()
        flag = False
        if item is None:
            q_out.put(None)
            stop_threads.set()
            break
        frame = deepcopy(item)
        x = obj_model(frame.to(dev))
        for predheads in x:
            print(torch.sum(predheads))
            if torch.sum(predheads) == 0:
                print("NaN in prediction")
                flag = True
                break
            # print(f"InferencePass predheads {predheads.shape}")
        if flag == False:
            y = postprocess(x)
            output = non_max_suppression(
                y,
                conf_thres=0.2,
                iou_thres=0.2,
                max_det=300,
                nc=len(cls),
                multi_label=False,
            )  # bbox xyxy, score, cls, nmask

            framedraw = plot_dod_batch(frame, output)
            for i in range(framedraw.shape[0]):
                cv2.imshow(
                    "result", framedraw[i].to("cpu").permute(1, 2, 0).numpy().copy()
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_threads.set()
                    break
            q_out.put(framedraw)
    q_out.put(None)


def LoadModel():
    """
    Load the object detection model.

    Returns:
        obj_model (DODv2): The object detection model.
    """
    obj_model = DODv2(
        nclasses=len(params.classes),
        reg_max=params.params["reg_max"],
        device=params.params["device"],
    )
    obj_model.load_state_dict(
        torch.load(f"{params.root}/DODcicd/models/DODv2_optimized.pth")
    )
    if str(params.params["device"]) == "cuda":
        obj_model.to(params.params["device"])
    else:
        obj_model.to("cpu")
    for _, param in obj_model.named_parameters():
        param.requires_grad = False
    obj_model.eval()
    print(
        summary(
            obj_model,
            torch.zeros((1, 3, 320, 320)).to(params.params["device"]),
            show_input=True,
        )
    )
    print(type(obj_model))
    return obj_model


# ---------

# --------- MAIN
if __name__ == "__main__":
    # threads for preiodically clean queues
    thread_clean_q = Thread(target=clear_q, args=(q, 4))
    thread_clean_b = Thread(target=clear_b, args=(b, 2))
    thread_clean_pred = Thread(target=clear_pred, args=(pred, 1))

    # threads for video aquisition and batch construction
    thread_vid = Thread(target=ReadFrame, args=(q,))
    thread_batch = Thread(
        target=BatchConstruct,
        args=(
            q,
            b,
        ),
    )

    classes = params.classes
    device = params.params["device"]
    model = LoadModel()
    inference = Inference(
        nclasses=len(classes),
        stride=torch.tensor(
            [
                8,
            ]
        ),
        reg_max=params.params["reg_max"],
        device=device,
    )

    # threads for inference and visualization
    thread_obj = Thread(
        target=InferencePass,
        args=(
            b,
            pred,
            model,
            inference,
            classes,
            device,
        ),
    )

    # set threads as daemon
    thread_clean_q.daemon = True
    thread_clean_b.daemon = True
    thread_clean_pred.daemon = True
    thread_vid.daemon = True
    thread_batch.daemon = True
    thread_obj.daemon = True

    # start threads
    thread_clean_q.start()
    thread_clean_b.start()
    thread_clean_pred.start()
    thread_vid.start()
    thread_batch.start()
    thread_obj.start()

    # append threads
    threads = []
    threads.append(thread_clean_q)
    threads.append(thread_clean_b)
    threads.append(thread_clean_pred)
    threads.append(thread_vid)
    threads.append(thread_batch)
    threads.append(thread_obj)

    # create a window to call program end
    safe = np.zeros((100, 100), np.uint8)
    while True:
        n1 = dt.datetime.now()
        try:
            cv2.imshow("`ESC` to exit", safe)
        except (ValueError, IOError) as err:
            logging.warning("An error occurred: %s", err)
            # Signal all threads to stop
            stop_threads.set()
        n2 = dt.datetime.now()
        elp = (n2 - n1).microseconds / 1000
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_threads.set()
            break
    # wait for threads to finish
    for t in threads:
        t.join()
    print("All threads finished")
    cv2.destroyAllWindows()
    sys.exit()

# --------
