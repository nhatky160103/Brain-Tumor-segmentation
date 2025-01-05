"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - A customized U-Net model
    Hua Wang
"""

from unet.mylib import *
from torchinfo import summary


class Unet(nn.Module):
    """
    A U-Net model
    """
    def __init__(self, in_channel):
        """
        Constructor -- create a new instance of Unet
        :param self: the current object
        :param in_channel: channel size of the input tensor
        """
        super(Unet, self).__init__()

        # layers
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.down1 = DoubleConv(in_channel, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)
        self.up6 = Up(512, 256)
        self.up7 = Up(256, 128)
        self.up8 = Up(128, 64)
        self.up9 = Up(64, 32)

        # output a single channel (binary)
        self.up10 = nn.ConvTranspose2d(32, 1, stride=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_input):
        """
        Method -- The forward function passes the input tenser through the
                    U-Net and returns the predicted result
        :param in_input: tensor, [batch size, in_channel, H, W]
        :return: tensor, predicted result, [batch size, 1, H, W]
        """
        im_down1 = self.down1(in_input)
        im_down1_pooled = self.max_pool(im_down1)
        im_down2 = self.down2(im_down1_pooled)
        im_down2_pooled = self.max_pool(im_down2)
        im_down3 = self.down3(im_down2_pooled)
        im_down3_pooled = self.max_pool(im_down3)
        im_down4 = self.down4(im_down3_pooled)
        im_down4_pooled = self.max_pool(im_down4)
        im_down5 = self.down5(im_down4_pooled)  # max pooling is not followed
        im_up6 = self.up6(im_down5, im_down4)
        im_up7 = self.up7(im_up6, im_down3)
        im_up8 = self.up8(im_up7, im_down2)
        im_up9 = self.up9(im_up8, im_down1)
        im_output = self.up10(im_up9)
        return self.sigmoid(im_output)


def main():
    # create a U-Net model
    unet = Unet(3)
    # show the information
    summary(unet, input_size=(1, 3, 256, 256))


if __name__ == '__main__':
    main()
