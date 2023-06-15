#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from pathlib import Path
import argparse


"""
Created on Friday Mar 13 2020
Modified on Saturday Mar 21 2020
@authors: Alan Preciado, Santosh Muthireddy
"""


def plot_loss_acc(
    parent_dataset: str, source, target, no_epochs, include_no_adapt: bool
):
    # specify path where in log folder where training logs are saved
    pkldir = os.path.join(
        "logs",
        parent_dataset,
        source + "_to_" + target,
        str(no_epochs) + "_epochs_128_s_128_t_batch_size",
    )

    if Path(pkldir).is_dir():
        print("directory with pkl files is:", pkldir)

    else:
        print(f"{pkldir} does not exist, must train model")
        return None

    # load dictionaries with log information
    path_adapt_log = [
        pkldir + "/adaptation_training_statistic.pkl",
        pkldir + "/adaptation_testing_s_statistic.pkl",
        pkldir + "/adaptation_testing_t_statistic.pkl",
    ]

    path_no_adapt_log = [
        pkldir + "/no_adaptation_training_statistic.pkl",
        pkldir + "/no_adaptation_testing_s_statistic.pkl",
        pkldir + "/no_adaptation_testing_t_statistic.pkl",
    ]

    print(">>>Loading pkl files<<<")
    adapt_training_dict = pickle.load(open(path_adapt_log[0], "rb"))
    adapt_testing_source_dict = pickle.load(open(path_adapt_log[1], "rb"))
    adapt_testing_target_dict = pickle.load(open(path_adapt_log[2], "rb"))

    # no_adapt_training_dict = pickle.load(open(path_no_adapt_log[0], "rb"))

    print(np.shape(adapt_testing_source_dict))

    if include_no_adapt:
        no_adapt_testing_source_dict = pickle.load(open(path_no_adapt_log[1], "rb"))
        no_adapt_testing_target_dict = pickle.load(open(path_no_adapt_log[2], "rb"))
        print(np.shape(no_adapt_testing_source_dict))

    print(">>>pkl files loaded correctly<<<")

    # create dictionary structures for adaptation and no-adaptation results

    # (w ddc loss)
    adaptation = {
        "classification_loss": [],
        "ddc_loss": [],
        "source_accuracy": [],
        "target_accuracy": [],
    }

    # (w/o ddc loss)
    no_adaptation = {"source_accuracy": [], "target_accuracy": []}

    # get average coral and classification loss for steps in each epoch
    # get accuracy obtained in each epoch
    for epoch_idx in range(len(adapt_training_dict)):  # epoch
        ddc_loss = 0
        class_loss = 0

        for step_idx in range(len(adapt_training_dict[epoch_idx])):
            ddc_loss += adapt_training_dict[epoch_idx][step_idx]["ddc_loss"]
            class_loss += adapt_training_dict[epoch_idx][step_idx][
                "classification_loss"
            ]

        # store average losses in general adaptation dictionary
        adaptation["classification_loss"].append(
            class_loss / len(adapt_training_dict[epoch_idx])
        )
        adaptation["ddc_loss"].append(ddc_loss / len(adapt_training_dict[epoch_idx]))
        adaptation["source_accuracy"].append(
            adapt_testing_source_dict[epoch_idx]["accuracy %"]
        )
        adaptation["target_accuracy"].append(
            adapt_testing_target_dict[epoch_idx]["accuracy %"]
        )

        if include_no_adapt:
            # store accuracies in no-adaptation dictionary
            no_adaptation["source_accuracy"].append(
                no_adapt_testing_source_dict[epoch_idx]["accuracy %"]
            )
            no_adaptation["target_accuracy"].append(
                no_adapt_testing_target_dict[epoch_idx]["accuracy %"]
            )

    # plot accuracies for test data in source and target domains
    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("classification accuracy (%)", fontsize=15)

    plt.plot(
        adaptation["target_accuracy"],
        label="test acc. w/ ddc loss",
        marker="*",
        markersize=8,
    )
    plt.plot(
        adaptation["source_accuracy"],
        label="training acc. w/ ddc loss",
        marker="^",
        markersize=8,
    )

    if include_no_adapt:
        plt.plot(
            no_adaptation["target_accuracy"],
            label="test acc. w/o ddc loss",
            marker=".",
            markersize=8,
        )
        plt.plot(
            no_adaptation["source_accuracy"],
            label="training acc. w/o ddc loss",
            marker="+",
            markersize=8,
        )

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.suptitle(source + "_to_" + target)
    fig.savefig(
        os.path.join(pkldir, source + "_to_" + target + "_test_train_accuracies.jpg")
    )

    # plot losses for test data in source and target domains
    fig = plt.figure(figsize=(8, 6), dpi=100)
    fig.show()

    plt.xlabel("epochs", fontsize=15)
    plt.ylabel("loss", fontsize=15)

    plt.plot(
        adaptation["classification_loss"],
        label="classification_loss",
        marker="*",
        markersize=8,
    )
    plt.plot(adaptation["ddc_loss"], label="ddc_loss", marker=".", markersize=8)

    plt.legend(loc="best")
    plt.grid()
    plt.show()
    fig.suptitle(source + "_to_" + target)
    fig.savefig(os.path.join(pkldir, source + "_to_" + target + "_train_losses.jpg"))


def main():
    parser = argparse.ArgumentParser(description="plots DeepCORAL")

    parser.add_argument("--source", default="amazon", type=str, help="source data")

    parser.add_argument("--target", default="dslr", type=str, help="target data")

    parser.add_argument("--no_epochs", default=100, type=int)

    parser.add_argument("--include_no_adapt", action="store_true", help="TODO")

    parser.add_argument(
        "--parent_dataset",
        type=str,
        help="Choose from available alternative datasets such as officehome. Default is office31.",
        default="office31",
    )

    args = parser.parse_args()

    plot_loss_acc(
        parent_dataset=args.parent_dataset,
        source=args.source,
        target=args.target,
        no_epochs=args.no_epochs,
        include_no_adapt=args.include_no_adapt,
    )


if __name__ == "__main__":
    main()
