import torch
from torch import nn


def loss_l1(out, target):

    return nn.functional.l1_loss(out, target)


def loss_sl1(out, target):

    return nn.functional.smooth_l1_loss(out, target)


def loss_l2(out, target):

    return nn.functional.mse_loss(out, target)

def loss_crossEn(out, target):

    return nn.functional.binary_cross_entropy(out, target)
