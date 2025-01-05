import random
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import cv2

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['text.color'] = '#DDDDDD'


def display_image_channels(image,n_slice ,title='Image Channels'):
    channel_names = ['Flair','T1', 'T1ce', 'T2']
    fig, axes = plt.subplots(1, 4, figsize=(6, 3))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :, n_slice]  # Transpose the array to display the channel
        ax.imshow(channel_image, cmap='magma')
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
    plt.show()


class ConvertToSingleChannelFromMultiChannel:
    def __call__(self, mask):
        image = torch.zeros(mask.shape[1:], dtype=torch.int64) if isinstance(mask, torch.Tensor) else np.zeros(
            mask.shape[1:], dtype=np.int64)

        # Áp dụng gán nhãn theo thứ tự ưu tiên từ cao đến thấp
        image[(mask[0] == 1) & (mask[1] == 1) & (mask[2] == 1)] = 3  # label 4
        image[(mask[0] == 1) & (mask[1] == 1) & (mask[2] == 0)] = 1  # label 1
        image[(mask[0] == 0) & (mask[1] == 1) & (mask[2] == 0)] = 2  # label 2

        return image


def display_mask_channels_as_rgb(mask, n_slice,  title='Mask Channels as RGB'):
    '''

    :param mask: (3, 128, 128, 128)
    :param title:
    :return: show the mask
    '''

    mask = ConvertToSingleChannelFromMultiChannel()(mask)
    mask = mask[..., n_slice]

    channel_names = ['Necrotic (NEC)', 'Edema (ED)', 'Tumour (ET)']
    fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    for idx, ax in enumerate(axes):
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        rgb_mask[..., idx] = (mask == idx + 1) * 255
        ax.imshow(rgb_mask)

        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.suptitle(title, fontsize=20, y=0.93)
    plt.tight_layout()
    plt.show()


def overlay_masks_on_image(image, mask, title='Brain MRI with Tumour Masks Overlay'):
    t1_image = image[0]
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())
    rgb_image = np.repeat(t1_image_normalized[..., np.newaxis], 3, axis=-1)  # Chuyển thành ảnh RGB

    colors = np.array([
        [0, 0, 0],  # 0 -> đen (background)
        [255, 0, 0],  # 1 -> đỏ
        [0, 255, 0],  # 2 -> xanh lá
        [0, 0, 255]  # 3 -> xanh dương
    ], dtype=np.uint8)

    rgb_mask = colors[mask]

    overlay_image = np.where(rgb_mask.any(axis=-1, keepdims=True), rgb_mask, rgb_image * 255).astype(np.uint8)

    plt.figure(figsize=(5, 5))
    plt.imshow(overlay_image)
    plt.title(title, fontsize=18, y=1.02)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

