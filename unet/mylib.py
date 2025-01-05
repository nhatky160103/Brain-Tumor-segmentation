"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - Customized tool layers
    Hua Wang
"""

import torch
from torch import nn
from torch import cat


class DoubleConv(nn.Module):
    """
    double_conv layer contains two convolutional layer
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(self, in_channel, out_channel):
        """
        Constructor -- create a new DoubleConv layer
        :param in_channel: input channel size
        :param out_channel: output channel size
        """
        # inherited properties passed from nn.Module
        super(DoubleConv, self).__init__()
        # define mid channel
        self.mid_channel = out_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.double_conv = nn.Sequential(
            # out_channel means how many convolutional kernels are we using
            # padding = 'same' keeps image size (HxW)
            nn.Conv2d(self.in_channel, self.mid_channel,
                      kernel_size=3, padding='same'),
            # batch normalization making sure values
            # in feature map follows normal distribution
            nn.BatchNorm2d(self.mid_channel),
            # an activation layer after each convolutional layer.
            # Turn on inplace to save memory
            nn.ReLU(inplace=True),

            nn.Conv2d(self.mid_channel, self.out_channel,
                      kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_layer):
        """
        Method -- The forward function passes input_layer through
                    the double convolutional layer
        :param input_layer: tensor [batch size, in_channel, H, W]
        :return: tensor [batch size, out_channel, H, W]
        """
        output_layer = self.double_conv(input_layer)
        return output_layer


class Up(nn.Module):
    """
    Up layer containing an inverse convolution layer
    and a double convolution layer
    ConvTranspose2d -> DoubleConv
    """

    def __init__(self, in_channel, out_channel):
        """
        Constructor -- create a new Up layer
        :param in_channel: input channel size
        :param out_channel: output channel size
        """
        # out_channel is 2 * in_channel
        super(Up, self).__init__()
        self.mid_channel = out_channel
        # an up-conv layer
        # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        self.up = nn.ConvTranspose2d(in_channel,
                                     self.mid_channel,
                                     stride=2,
                                     kernel_size=2)
        # a DoubleConv layer
        # input channel size is 2 * mid channel size
        # because we need to concatenate
        self.double_conv = DoubleConv(self.mid_channel * 2, out_channel)

    def forward(self, input_layer1, input_layer2):
        """
        Method -- The forward function concatenate input_layer1 and
                  input_layer2, and passes the result layer through
                  the double_conv layer
        :param input_layer1: tensor [batch size, in_channel, H, W]
        :param input_layer2: tensor [batch size, in_channel, H, W]
        :return: tensor [batch size, out_channel, H, W]
        """
        input_layer1 = self.up(input_layer1)
        input_layer = cat((input_layer1, input_layer2), dim=1)  # axis: channel
        output_layer = self.double_conv(input_layer)
        return output_layer


class MyLoss(nn.Module):
    """
    MyLoss layer calculate the loss of the prediction layer
    Two Loss Mode: Dice loss or IOU loss
    """

    def __init__(self, dice_loss_mode: bool = True, smooth: float = 0.01):
        """
        Constructor -- create a new MyLoss layer
        :param dice_loss_mode: bool, use dice loss
        :param smooth: float
        """
        super(MyLoss, self).__init__()
        self.dice_loss_mode = dice_loss_mode
        self.smooth = smooth
        return

    def forward(self, pred, target):
        """
        Method -- The forward function calculate the loss between
                    the pred and the target using a specific loss mode.

        :param pred: tensor [batch size, in_channel, H, W]
        :param target: tensor [batch size, in_channel, H, W]
        :return: tensor
        """
        if self.dice_loss_mode:
            # flatten pred and target
            pred_flattened = pred.reshape(-1)
            target_flattened = target.reshape(-1)

            # intersection
            intersect = torch.dot(pred_flattened, target_flattened)

            # sum
            sum_two = torch.sum(pred_flattened) + torch.sum(target_flattened)

            # dice_loss = 2 * |A∩B| / |A∪B|
            loss = - (2 * intersect + self.smooth) / (sum_two + self.smooth)
        else:
            # intersection
            intersect = torch.sum(pred * target)

            # union
            union = torch.ceil((pred + target) / 2)

            # iou_loss = |A∩B| / |A∪B|
            loss = - (intersect + self.smooth) / (torch.sum(union) +
                                                  self.smooth)
        return loss
