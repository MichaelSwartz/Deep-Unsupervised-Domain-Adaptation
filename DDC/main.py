#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import argparse
import warnings
import math
from tqdm import tnrange
import torch
from torch.autograd import Variable

warnings.filterwarnings("ignore")

from train import train
from test import test
from loss import DDC_loss
from utils import load_pretrained_AlexNet, save_log, save_model, load_model, load_resnet
from dataloader import get_office_dataloader
from model import DDCNet, AlexNet


# set model hyperparameters (paper page 5)
CUDA = True if torch.cuda.is_available() else False
learning_rate = 1e-3
L2_DECAY = 5e-4
MOMENTUM = 0.9


def step_decay(epoch, learning_rate):
    """
    Schedule step decay of learning rate with epochs.
    """
    initial_learning_rate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    learning_rate = initial_learning_rate * math.pow(
        drop, math.floor((1 + epoch) / epochs_drop)
    )

    return learning_rate


def main():
    """
    This method puts all the modules together to train DDCNet for image
    classification. It uses a MMD loss in the last classification layer for
    domain adaptation.

    Paper: https://arxiv.org/abs/1412.3474
    """
    parser = argparse.ArgumentParser(description="domain adaptation w MMD")

    parser.add_argument(
        "--epochs", default=10, type=int, help="number of training epochs"
    )

    parser.add_argument(
        "--batch_size_source", default=128, type=int, help="batch size of source data"
    )

    parser.add_argument(
        "--batch_size_target", default=56, type=int, help="batch size of target data"
    )

    parser.add_argument(
        "--name_source",
        default="amazon",
        type=str,
        help="name of source dataset (default amazon)",
    )

    parser.add_argument(
        "--name_target",
        default="webcam",
        type=str,
        help="name of source dataset (default webcam)",
    )

    parser.add_argument(
        "--num_classes",
        default=31,
        type=int,
        help="no. classes in dataset (default 31)",
    )

    parser.add_argument(
        "--load_model",
        default=None,
        type=None,
        help="load pretrained model (default None)",
    )

    parser.add_argument(
        "--adapt_domain",
        action="store_true",
        help="argument to compute coral loss (default False)",
    )

    parser.add_argument(
        "--parent_dataset",
        type=str,
        help="Choose from available alternative datasets such as officehome. Default is office31.",
        default="office31",
    )

    args = parser.parse_args()

    # create dataloaders (Amazon --> source, Webcam --> target)
    print("creating source/target dataloaders...")
    print("source data:", args.name_source)
    print("target data:", args.name_target)

    source_loader = get_office_dataloader(
        sub_dataset=args.name_source,
        parent_dataset=args.parent_dataset,
        batch_size=args.batch_size_source,
    )

    target_loader = get_office_dataloader(
        sub_dataset=args.name_target,
        parent_dataset=args.parent_dataset,
        batch_size=args.batch_size_target,
    )

    # define DDCNet model
    model = DDCNet(num_classes=args.num_classes)

    # move to CUDA if available
    if CUDA:
        model = model.cuda()
        print("using cuda...")

    # load pre-trained model or pre-trained AlexNet
    if args.load_model is not None:
        if args.load_model == "resnet":
            load_resnet(model)
        else:
            load_model(model, args.load_model)  # contains path to model params
    else:
        load_pretrained_AlexNet(model.sharedNetwork, progress=True)

    print("model type:", type(model))

    # store statistics of train/test
    training_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    # start training over epochs
    print("adapt domain:", args.adapt_domain)
    print("running training for {} epochs...".format(args.epochs))

    for epoch in tnrange(0, args.epochs):
        log_interval = 10
        LEARNING_RATE = step_decay(epoch, learning_rate)
        print("Current learning rate:", LEARNING_RATE)

        optimizer = torch.optim.SGD(
            [
                {"params": model.sharedNetwork.parameters()},
                {"params": model.bottleneck.parameters(), "lr": LEARNING_RATE},
                {"params": model.fc8.parameters(), "lr": LEARNING_RATE},
            ],
            lr=LEARNING_RATE / 10,
            momentum=MOMENTUM,
            weight_decay=L2_DECAY,
        )

        # compute lambda value from paper (eq 6)
        if args.adapt_domain:
            lambda_factor = (epoch + 1) / args.epochs  # adaptation (w/ coral loss)

        else:
            lambda_factor = 0  # no adaptation (w/o coral loss)

        # run batch trainig at each epoch (returns dictionary with epoch result)
        result_train = train(
            model,
            source_loader,
            target_loader,
            optimizer,
            epoch + 1,
            lambda_factor,
            CUDA,
        )

        # print log values
        print(
            "[EPOCH] {}: Classification loss: {:.6f}, DDC loss: {:.6f}, Total_Loss: {:.6f}".format(
                epoch + 1,
                sum(
                    row["classification_loss"] / row["total_steps"]
                    for row in result_train
                ),
                sum(row["ddc_loss"] / row["total_steps"] for row in result_train),
                sum(row["total_loss"] / row["total_steps"] for row in result_train),
            )
        )

        training_statistic.append(result_train)

        # test classification accuracy on both datasets
        test_source = test(model, source_loader, epoch, CUDA)
        test_target = test(model, target_loader, epoch, CUDA)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)

        print(
            "[Test Source]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch + 1,
                test_source["average_loss"],
                test_source["correct_class"],
                test_source["total_elems"],
                test_source["accuracy %"],
            )
        )

        print(
            "[Test Target]: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                epoch + 1,
                test_target["average_loss"],
                test_target["correct_class"],
                test_target["total_elems"],
                test_target["accuracy %"],
            )
        )

    # save log results
    if args.adapt_domain:
        print("saving training with adaptation...")
        save_log(training_statistic, "adaptation_training_statistic.pkl")
        save_log(testing_s_statistic, "adaptation_testing_s_statistic.pkl")
        save_log(testing_t_statistic, "adaptation_testing_t_statistic.pkl")
        save_model(model, "adaptation_checkpoint.tar")

    else:
        print("saving training without adaptation...")
        save_log(training_statistic, "no_adaptation_training_statistic.pkl")
        save_log(testing_s_statistic, "no_adaptation_testing_s_statistic.pkl")
        save_log(testing_t_statistic, "no_adaptation_testing_t_statistic.pkl")
        save_model(model, "no_adaptation_checkpoint.tar")


if __name__ == "__main__":
    main()
