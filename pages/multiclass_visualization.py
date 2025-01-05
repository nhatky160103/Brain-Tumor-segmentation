import os
import zipfile
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from Segformer3d.model import segformer3d_get_predict
from segnet.infer import segresnet_get_predict
from utils import load_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

st.title("Segformer3D Model Prediction for Brain MRI")

model_list = ['segresnet_1', 'segresnet_2','segresnet_origin' ,'segformer3d']
model_name = st.selectbox("Select model", model_list, index=0)

uploaded_zip = st.file_uploader("Upload a zip file containing MRI data", type=["zip"])

n_slice = st.slider("Select a slice", min_value=0, max_value=155, value=55)

# Tạo lựa chọn colormap cho ảnh gốc, label và prediction
cmap_list = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'spring', 'summer', 'autumn', 'winter']
cmap_choice_image = st.selectbox("Select colormap for Original Image", cmap_list, index=0)
cmap_choice_label = st.selectbox("Select colormap for Label", cmap_list, index=1)
cmap_choice_pred = st.selectbox("Select colormap for Prediction", cmap_list, index=2)


def display_image_channels(image, n_slice, title='Image Channels', cmap='magma'):
    channel_names = ['Flair', 'T1', 'T1ce', 'T2']
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for idx, ax in enumerate(axes.flatten()):
        channel_image = image[idx, :, :, n_slice]
        ax.imshow(channel_image, cmap=cmap)
        ax.axis('off')
        ax.set_title(channel_names[idx])
    plt.tight_layout()
    plt.suptitle(title, fontsize=20, y=1.03)
    st.pyplot(fig)


def display_label_and_prediction(gt, pred, label_back, pred_back, n_slice, cmap_label, cmap_pred):
    # Tạo một figure với 3 subplots (3 cho Label và 3 cho Prediction)
    fig, axes = plt.subplots(2, 3, figsize=(18, 6))  # 2 hàng, 3 cột

    # Hiển thị 3 Label images
    for i in range(3):  # Loop qua 3 channel của ảnh label
        axes[0, i].imshow(gt[i, :, :, n_slice], cmap=cmap_label)  # Chọn channel i của ảnh label
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Label Channel {i + 1}")

    # Hiển thị 3 Prediction images
    for i in range(3):  # Loop qua 3 channel của ảnh prediction
        axes[1, i].imshow(pred[i, :, :, n_slice], cmap=cmap_pred)  # Chọn channel i của ảnh prediction
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Prediction Channel {i + 1}")

    # Hiển thị label_back và pred_back (ảnh 3 kênh từ label_back và pred_back)
    fig_back, axes_back = plt.subplots(1, 2, figsize=(12, 6))
    axes_back[0].imshow(label_back[:, :, n_slice], cmap=cmap_label)  # Ảnh label_back chỉ có 1 kênh (3D)
    axes_back[0].axis('off')
    axes_back[0].set_title("Label Back")

    axes_back[1].imshow(pred_back[:, :, n_slice], cmap=cmap_pred)  # Ảnh pred_back chỉ có 1 kênh (3D)
    axes_back[1].axis('off')
    axes_back[1].set_title("Prediction Back")

    plt.tight_layout()
    st.pyplot(fig)
    st.pyplot(fig_back)


def display_overlay(image, label_back, n_slice):
    t1_image = image[0][:, :, n_slice]
    label_slice = label_back[:, :, n_slice]
    label_slice[label_slice == 4] = 3
    t1_image_normalized = (t1_image - t1_image.min()) / (t1_image.max() - t1_image.min())

    rgb_image = np.repeat(t1_image_normalized[..., np.newaxis], 3, axis=-1)

    colors = np.array([
        [0, 0, 0],  # 0 -> Đen (background)
        [255, 0, 0],  # 1 -> Đỏ
        [0, 255, 0],  # 2 -> Vàng
        [0, 0, 255]  # 4 -> Xanh lá
    ], dtype=np.uint8)

    rgb_label = colors[label_slice]

    overlay_image = np.where(rgb_label.any(axis=-1, keepdims=True), rgb_label, rgb_image * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    axes.imshow(overlay_image)
    axes.set_title(f"Overlay for slice {n_slice}", fontsize=18)
    axes.axis('off')

    plt.tight_layout()
    st.pyplot(fig)


model = load_model(model_name)

if uploaded_zip is not None:
    # Tạo thư mục tạm thời để giải nén ZIP
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        temp_dir = './temp_mri_folder'
        os.makedirs(temp_dir, exist_ok=True)
        zip_ref.extractall(temp_dir)

    case_name = os.listdir(temp_dir)[0]
    print(case_name)
    image = None
    gt = None
    pred = None
    label_back= None
    pred_back = None

    if model_name == 'segformer3d':
        image, gt, pred, label_back, pred_back = segformer3d_get_predict(model, case_name)
    elif model_name in ['segresnet_1', 'segresnet_2', 'segresnet_origin']:
        image, gt, pred, label_back, pred_back  = segresnet_get_predict(model, case_name, model_name)
    else:
        print('please select model')


    display_image_channels(image, n_slice, title="Original Image Channels", cmap=cmap_choice_image)

    display_label_and_prediction(gt, pred, label_back, pred_back, n_slice, cmap_choice_label, cmap_choice_pred)

    display_overlay(image, label_back, n_slice)

