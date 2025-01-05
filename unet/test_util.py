"""
    CS5001 Fall 2022
    Final Project: Brain Tumor Segmentation
    - Test functions in util.py
    Hua Wang
"""
from util import *
import unittest
import pandas as pd
import torch


class UtilTest(unittest.TestCase):
    def test_adjust_data(self):
        """
        Test the adjust_data function
        :return: none
        """
        img = pd.DataFrame([[128, 255], [64, 0]])
        mask = pd.DataFrame([[255, 0], [0, 255]])
        img_result = pd.DataFrame([[128/255, 1.0], [64/255, 0.0]])
        mask_result = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]])
        img, mask = adjust_data(img, mask)
        self.assertTrue(img.equals(img_result))
        self.assertTrue(mask.equals(mask_result))

    def test_iou_score(self):
        """
        Test the iou_score function with smooth=0
        :return: none
        """
        pred = torch.tensor([[1., 1.], [1., 0.]])
        target = torch.tensor([[0., 0.], [1., 1.]])
        iou = iou_score(pred, target, smooth=0.0)
        self.assertEqual(iou, 0.25)

    def test_dice_score(self):
        """
        Test the dice_score function with smooth=0
        :return: none
        """
        pred = torch.tensor([[1., 1.], [1., 0.]])
        target = torch.tensor([[0., 0.], [1., 1.]])
        iou = dice_score(pred, target, smooth=0.0)
        self.assertEqual(iou, 0.4)


if __name__ == '__main__':
    unittest.main(verbosity=4)
