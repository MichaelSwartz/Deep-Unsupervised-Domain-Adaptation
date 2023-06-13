#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import DIRECTORIES


"""
Created on Saturday Feb 22 2020

@authors: Alan Preciado, Santosh Muthireddy
"""


def get_mean_std_dataset(root_dir):
    """
    Function to compute mean and std of image dataset.
    Move batch_size param according to memory resources.
    retrieved from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """

    # data_domain = "amazon"
    # path_dataset = "datasets/office/%s/images" % data_domain

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # original image size 300x300 pixels
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    # set large batch size to get good approximate of mean, std of full dataset
    # batch_size: 4096, 2048
    data_loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)

    mean = []
    std = []

    for i, data in enumerate(data_loader, 0):
        # shape is (batch_size, channels, height, width)
        npy_image = data[0].numpy()

        # compute mean, std per batch shape (3,) three channels
        batch_mean = np.mean(npy_image, axis=(0, 2, 3))
        batch_std = np.std(npy_image, axis=(0, 2, 3))

        mean.append(batch_mean)
        std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(mean).mean(axis=0)  # average over batch averages
    std = np.array(std).mean(axis=0)  # average over batch stds

    values = {"mean": mean, "std": std}

    return values


def get_office_dataloader(
    sub_dataset: str, parent_dataset: str, batch_size: int, train: bool = True
):
    """
    Creates dataloader for the datasets in office datasetself.
    Uses get_mean_std_dataset() to compute mean and std along the
    color channels for the datasets in office.
    """
    try:
        root_dir = DIRECTORIES[parent_dataset] % sub_dataset
        print(f"Root dir: {root_dir}")
    except KeyError:
        raise Exception(f"{parent_dataset} not in config.py DIRECTORIES")

    # root_dir = "datasets/office/%s/images" % name_dataset
    # root_dir = "/content/drive/My Drive/office/%s/images" % name_dataset

    if parent_dataset == "office31":
        __datasets__ = ["amazon", "dslr", "webcam"]

        if sub_dataset not in __datasets__:
            raise ValueError("must introduce one of the three datasets in office")

        # Ideally compute mean and std with get_mean_std_dataset.py
        # https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/data_loader.py
        mean_std = {
            "amazon": {
                "mean": [0.7923, 0.7862, 0.7841],
                "std": [0.3149, 0.3174, 0.3193],
            },
            "dslr": {"mean": [0.4708, 0.4486, 0.4063], "std": [0.2039, 0.1920, 0.1996]},
            "webcam": {
                "mean": [0.6119, 0.6187, 0.6173],
                "std": [0.2506, 0.2555, 0.2577],
            },
        }
        mean = mean_std[sub_dataset]["mean"]
        std = mean_std[sub_dataset]["std"]

    else:
        mean_std = get_mean_std_dataset(root_dir)
        mean = mean_std["mean"]
        std = mean_std["std"]

        print(f"Mean std for {root_dir}: {mean_std}")

    # compose image transformations
    data_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # retrieve dataset using ImageFolder
    # datasets.ImageFolder() command expects our data to be organized
    # in the following way: root/label/picture.png
    dataset = datasets.ImageFolder(root=root_dir, transform=data_transforms)

    # Dataloader is able to spit out random samples of our data,
    # so our model won’t have to deal with the entire dataset every time.
    # shuffle data when training
    dataset_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=4, drop_last=True
    )

    return dataset_loader
