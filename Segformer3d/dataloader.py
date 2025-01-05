import os
import cv2
import numpy as np
import torch
import monai.transforms as transforms
import nibabel as nib
from monai.transforms import EnsureType, Orientation
from monai.data import MetaTensor
from sklearn.preprocessing import MinMaxScaler
from monai.transforms import EnsureType

class ConvertBackToOriginalClasses(object):
    def __call__(self, multi_channel_img):
        # multi_channel_img có kích thước (3, 155, 155, 155)
        # Chuyển đổi lại về ảnh có lớp ban đầu
        tumor_core = multi_channel_img[0]
        total_tumor = multi_channel_img[1]
        enhance_tumor = multi_channel_img[2]

        # Non-enhance tumor (class 1)
        class_1 = np.logical_and(tumor_core, np.logical_not(enhance_tumor))  # Tumor core but not total tumor
        # Edema (class 2)
        class_2 = np.logical_and(total_tumor, np.logical_not(tumor_core))  # Total tumor but not enhance tumor
        # Enhance tumor (class 4)
        class_4 = enhance_tumor

        class_4 = class_4.astype(bool)

        # Kết hợp lại thành mảng gốc ban đầu với giá trị 1, 2, 4
        restored_label = np.zeros_like(tumor_core, dtype=int)


        restored_label[class_1] = 1  # Assign class 1
        restored_label[class_2] = 2  # Assign class 2
        restored_label[class_4] = 4  # Assign class 4


        return restored_label


class ConvertToMultiChannelBasedOnBrats2017Classes(object):
    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [(img == 4) | (img == 1), (img == 2) | (img == 4) | (img == 1), img == 4]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class Brats2017InferSingleSample:
    def __init__(self, root_dir: str, case_name: str):
        self.root_dir = root_dir
        self.case_name = case_name

    def normalize(self, x: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        return normalized_1D_array.reshape(x.shape)

    def orient(self, x: MetaTensor) -> MetaTensor:
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray) -> np.ndarray:
        return x[:, 40:195, 40:195, :]

    def get_modality_fp(self, mri_code: str = None) -> str:
        if mri_code:
            f_name = f"{self.case_name}_{mri_code}.nii"
        else:
            f_name = f"{self.case_name}.nii"
        return os.path.join(self.root_dir, self.case_name, f_name)

    def load_nifti(self, fp):
        nifti_data = nib.load(fp)
        return nifti_data.get_fdata(), nifti_data.affine

    def preprocess_brats_modality(self, data_fp: str, is_label: bool = False) -> np.ndarray:
        data, affine = self.load_nifti(data_fp)
        if is_label:
            data = data.astype(np.uint8)
            data = ConvertToMultiChannelBasedOnBrats2017Classes()(data)
        else:
            data = self.normalize(x=data)
            data = data[np.newaxis, ...]

        data = MetaTensor(x=data, affine=affine)
        data = self.orient(data)
        data = self.detach_meta(data)
        data = self.crop_brats2021_zero_pixels(data)
        return data

    def __call__(self):
        # Load and preprocess each modality
        flair_fp = self.get_modality_fp("flair",)
        t1w_fp = self.get_modality_fp("t1",)
        t1gd_fp = self.get_modality_fp("t1ce",)
        t2w_fp = self.get_modality_fp("t2",)
        label_fp = self.get_modality_fp("seg")

        Flair = self.preprocess_brats_modality(flair_fp, is_label=False)
        T1w = self.preprocess_brats_modality(t1w_fp, is_label=False)
        T1gd = self.preprocess_brats_modality(t1gd_fp, is_label=False)
        T2w = self.preprocess_brats_modality(t2w_fp, is_label=False)

        modalities = np.concatenate([Flair, T1w, T1gd, T2w], axis=0).astype(np.float32)
        if os.path.exists(label_fp):
            label = self.preprocess_brats_modality(label_fp, is_label=True)

        else:
            label = None
            
        data = {
            'image': modalities,
            'label': label
        }

        val_transform = transforms.Compose([
            transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
        ])
        transformed_data = val_transform(data)

        return transformed_data



if __name__ =="__main__":

    infer = Brats2017InferSingleSample(root_dir="Segformer3d/data", case_name="Brats17_2013_10_1")
    transformed_data = infer()

    modalities = transformed_data['image'].numpy()
    label = transformed_data['label'].numpy()

    print(modalities.shape)
    print(label.shape)

    print(modalities.max())
    print(modalities.min())
    print(np.unique(label))

    cv2.imshow('image', modalities[0, :, :, 60])
    cv2.waitKey(0)
    cv2.imshow('label1', label[0, :, :, 60].astype(np.uint8)*255)
    cv2.waitKey(0)
    cv2.imshow('label2', label[1, :, :, 60].astype(np.uint8)*255)
    cv2.waitKey(0)
    cv2.imshow('label3', label[2, :, :, 60].astype(np.uint8)*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()