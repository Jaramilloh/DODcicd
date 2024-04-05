import os
import torch
from main import (
    Resolution,
    Parameters,
    LoadImage,
    LoadModel,
    InferencePass,
    ResizePrediction,
)

input_dim = Resolution()
params = Parameters()


def test_LoadImage():
    imgfile = "dataset2_front_510.png"
    imgfile = f"{params.root}/DOD-ci-cd/images/{imgfile}"
    img = LoadImage(imgfile)
    zeros = torch.zeros([1, 3, 320, 320], dtype=torch.float32)
    assert zeros.shape == img.shape


def test_LoadModel():
    model = LoadModel()
    assert model.training == False


def test_InferencePass():
    imgfile = "dataset2_front_510.png"
    img = f"{params.root}/DOD-ci-cd/images/{imgfile}"
    predimg = InferencePass(img)
    assert predimg.shape == (params.params["img_size"], params.params["img_size"], 3)


def test_ResizePrediction():
    imgname = "dataset2_front_510.png"
    imgfile = f"{params.root}/DOD-ci-cd/images/{imgname}"
    predimg = InferencePass(imgfile)
    outfile = f"{params.root}/DOD-ci-cd/predictions/pred_{imgname}"
    if os.path.exists(outfile):
        os.remove(outfile)
    from PIL import Image

    input_image = Image.open(imgfile)
    width, height = input_image.size
    input_dim.width = width
    input_dim.height = height
    img = ResizePrediction(predimg, outfile)
    assert img.size == (input_dim.width, input_dim.height)
    assert os.path.exists(outfile)
    try:
        input_image = Image.open(outfile)
        input_image.verify()
        assert True
    except (IOError, SyntaxError) as e:
        assert False
