import os

import torch

from src.model import ConvModule  # pylint: disable=unused-import
from src.model import Bottleneck  # pylint: disable=unused-import
from src.model import C2f  # pylint: disable=unused-import
from src.model import SPPF  # pylint: disable=unused-import
from src.model import DetectionHead  # pylint: disable=unused-import
from src.model import ObjectDetectorV0


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


params = Parameters()

model = ObjectDetectorV0(
    nclasses=len(params.classes),
    reg_max=params.params["reg_max"],
    device=params.params["device"],
)
if str(params.params["device"]) == "cuda":
    model = torch.load(
        f"{params.root}/DODcicd/models/model_v0bsDepth_v1_finesse_63.pt"
    ).to(params.params["device"])
else:
    model = torch.load(
        f"{params.root}/DODcicd/models/model_v0bsDepth_v1_finesse_63.pt",
        map_location=torch.device("cpu"),
    ).to(params.params["device"])

torch.save(model.state_dict(), "optDOD.pth")
