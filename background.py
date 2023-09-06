import cv2
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import warnings
warnings.filterwarnings("ignore")

# Project imports
from data_loader_cache import normalize, im_reader, im_preprocess 
from models import *

# Helpers
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Download official weights
if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
    MODEL_PATH_URL = "https://drive.google.com/uc?id=1KyMpRjewZdyYfxHPYcd-ZbanIXtin0Sn"
    gdown.download(MODEL_PATH_URL, "saved_models/isnet.pth", use_cookies=False)

class GOSNormalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

transform =  transforms.Compose([GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])

def load_image(im_path, hypar):
    im, im_shp = im_preprocess(im_reader(im_path), hypar["cache_size"])
    im = torch.divide(im, 255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0)

def build_model(hypar, device):
    net = hypar["model"]
    if hypar["model_digit"] == "half":
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    net.to(device)
    if hypar["restore_model"] != "":
        net.load_state_dict(torch.load(f"{hypar['model_path']}/{hypar['restore_model']}", map_location=device))
        net.to(device)
    net.eval()
    return net

def predict(net, inputs_val, shapes_val, hypar, device):
    net.eval()
    if hypar["model_digit"] == "full":
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device)
    ds_val = net(inputs_val_v)[0]
    pred_val = ds_val[0][0, :, :, :]
    pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val - mi) / (ma - mi)
    if device == 'cuda':
        torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy() * 255).astype(np.uint8)

# Set Parameters
hypar = {}
hypar["model_path"] = "./saved_models"
hypar["restore_model"] = "isnet.pth"
hypar["model_digit"] = "full"
hypar["seed"] = 0
hypar["cache_size"] = [1024, 1024]
hypar["model"] = ISNetDIS()

# Build Model
net = build_model(hypar, device)

# Modified inference function
def inference(image: Image.Image):
    image_tensor, orig_size = load_image(image, hypar)
    mask = predict(net, image_tensor, orig_size, hypar, device)
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgba = image.copy()
    im_rgba.putalpha(pil_mask)
    return im_rgba, pil_mask
