#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage import io
import pickle
import matplotlib.pyplot as plt
import os
import torch
from torchvision.models import resnet50, ResNet50_Weights

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""


def load_pretrained_AlexNet(model, progress=True):
    # def alexnet(pretrained=False, progress=True, **kwargs):
    """
    AlexNet model architecture from the paper
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    progress (bool): If True, displays a progress bar of the download to stderr
    """

    __all_ = ["AlexNet", "alexnet", "Alexnet"]

    model_url = {
        "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    }

    print("loading pre-trained AlexNet...")
    state_dict = load_state_dict_from_url(model_url["alexnet"], progress=progress)
    model_dict = model.state_dict()

    # filter out unmatching dictionary
    # reference: https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(state_dict)
    model.load_state_dict(state_dict)
    print("loaded model correctly...")


def save_log(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
        print("[INFO] Object saved to {}".format(path))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("checkpoint saved in {}".format(path))


def load_model(model, path):
    """
    Loads trained network params in case AlexNet params are not loaded.
    """
    model.load_state_dict(torch.load(path))
    print("pre-trained model loaded from {}".format(path))


def load_resnet(model):
    print("loading pre-trained ResNet...")
    model.sharedNetwork = resnet50(weights=ResNet50_Weights.DEFAULT)
    print("loaded model correctly...")


def show_image(dataset, domain, image_class, image_name):
    """
    Plot images from given domain, class
    """
    image_file = io.imread(
        os.path.join("data", dataset, domain, "images", image_class, image_name)
    )
    plt.imshow(image_file)
    plt.pause(0.001)
    plt.figure()
