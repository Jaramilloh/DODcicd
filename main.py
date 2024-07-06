import os
import sys
import logging

# import time
import numpy as np
import torch
from pytorch_model_summary import summary
from PIL import Image

from src.inference import Inference, non_max_suppression
from src.ops import visualize_pred

from src.model import ConvModule  # pylint: disable=unused-import
from src.model import Bottleneck  # pylint: disable=unused-import
from src.model import C2f  # pylint: disable=unused-import
from src.model import SPPF  # pylint: disable=unused-import
from src.model import DetectionHead  # pylint: disable=unused-import
from src.model import DODv2

# Create a custom logger
LOGGER = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("file.log")
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


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    LOGGER.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class Resolution:
    def __init__(self):
        self.width = None
        self.height = None
        self.ratio = None


class Parameters:
    def __init__(self):
        self.cwd = os.getcwd()
        self.root = os.path.dirname(self.cwd)
        self.classes = {0: "fruit"}
        self.params = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "img_size": 320,
            "reg_max": 4,
        }


input_dim = Resolution()
params = Parameters()


def LoadImage(file_name):
    try:
        image = Image.open(file_name)
    except (Exception,):
        image = np.array(params.params["img_size"] * [0, 0, 0])
        LOGGER.error("Could not open image file %s", file_name)
        raise
    # calculate if wide or lenght dimensions should be resize into fixed lenght and preserve the aspect ratio
    width, height = image.size
    input_dim.width = width
    input_dim.height = height
    if width > height:
        ratio = params.params["img_size"] / width * 1.0
        new_width = int(width * ratio)
        new_height = int(params.params["img_size"])
    else:
        ratio = params.params["img_size"] / height * 1.0
        new_width = int(params.params["img_size"])
        new_height = int(height * ratio)
    # Resize the image to the desired size
    input_dim.ratio = ratio
    input_image = image.resize((new_width, new_height))
    # Convert the image into a numpy array, then convert it to torch tensor
    input_image = np.array(input_image)
    input_image = input_image[:, :, ::] / 255.0
    input_image = torch.tensor(input_image, dtype=torch.float32)
    input_image.permute(2, 0, 1)
    input_image = input_image.unsqueeze(0)
    input_image = input_image.permute(0, 3, 1, 2)
    # LOGGER.info("Input image shape: %s", str(input_image.shape))
    return input_image


def LoadModel():
    model = DODv2(
        nclasses=len(params.classes),
        reg_max=params.params["reg_max"],
        device=params.params["device"],
    )
    model.load_state_dict(
        torch.load(f"{params.root}/DODcicd/models/DODv2_optimized.pth")
    )
    if str(params.params["device"]) == "cuda":
        model.to(params.params["device"])
    else:
        model.to("cpu")
    for _, param in model.named_parameters():
        param.requires_grad = False
    model.eval()
    print(
        summary(
            model,
            torch.zeros((1, 3, 320, 320)).to(params.params["device"]),
            show_input=True,
        )
    )
    print(type(model))

    return model


def InferencePass(filename, model=LoadModel()):
    imagec = LoadImage(filename)
    print(imagec.shape)
    print(input_dim.ratio)
    pred = model(imagec.to(params.params["device"]))
    inference = Inference(
        nclasses=len(params.classes),
        stride=torch.tensor(
            [
                8,
            ]
        ),
        reg_max=params.params["reg_max"],
        device=params.params["device"],
    )
    y = inference(pred)
    output = non_max_suppression(
        y,
        conf_thres=0.2,
        iou_thres=0.2,
        max_det=300,
        nc=len(params.classes),
        multi_label=False,
    )  # bbox xyxy, score, cls, nmask
    predimg = visualize_pred(imagec, output)
    return predimg


def ResizePrediction(imgage, filename):
    imgage = Image.fromarray(imgage)
    imgage = imgage.resize((input_dim.width, input_dim.height))
    imgage.save(filename)
    return imgage


imgfile = "dataset2_front_510.png"
img = f"{params.root}/DODcicd/images/{imgfile}"
outfile = f"{params.root}/DODcicd/predictions/pred_{imgfile}"

if __name__ == "__main__":
    predicted_image = InferencePass(img)
    predicted_image = ResizePrediction(predicted_image, outfile)
