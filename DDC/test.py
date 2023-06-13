#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from tqdm import tnrange
from torch.autograd import Variable


def test(model, data_loader, epoch, cuda):
    """
    Computes classification accuracy of (labeled) data using cross-entropy.
    """
    # eval() it indicates the model that nothing new is
    # to be learnt and the model is used for testing
    model.eval()

    test_loss = 0
    correct_class = 0

    # go over dataloader batches, labels
    for data, label in data_loader:
        if cuda:
            data, label = data.cuda(), label.cuda()

        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
        data, label = Variable(data, volatile=True), Variable(label)
        output, _ = model(data, data)  # just use one ouput of DeepCORAL

        # sum batch loss when computing classification
        test_loss += torch.nn.functional.cross_entropy(
            output, label, size_average=False
        ).item()
        # test_loss += torch.nn.functional.cross_entropy(output, label, size_average=False).data[0]

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct_class += pred.eq(label.data.view_as(pred)).cpu().sum()

    # compute test loss as correclty classified labels divided by total data size
    test_loss = test_loss / len(data_loader.dataset)

    # return dictionary containing info of each epoch
    return {
        "epoch": epoch + 1,
        "average_loss": test_loss,
        "correct_class": correct_class.item(),
        "total_elems": len(data_loader.dataset),
        "accuracy %": (100.0 * correct_class / len(data_loader.dataset)).item(),
    }
