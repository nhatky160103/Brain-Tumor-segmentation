import os
import torch
import nibabel
import numpy as np
from tqdm import tqdm
from monai.data import MetaTensor
from monai.transforms import Orientation, EnsureType
from sklearn.preprocessing import MinMaxScaler


class ConvertToMultiChannelBasedOnBrats2017Classes(object):
    def __call__(self, img):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        result = [(img == 2) | (img == 3), (img == 2) | (img == 3) | (img == 1), img == 3]
        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class Brats2017Task1Preprocess:
    def __init__(self, root_dir: str, train_folder_name: str = "train", save_dir: str = "../BraTS2017_Training_Data"):
        self.train_folder_dir = os.path.join(root_dir, train_folder_name)
        label_folder_dir = os.path.join(root_dir, train_folder_name, "labelsTr")
        assert os.path.exists(self.train_folder_dir)
        assert os.path.exists(label_folder_dir)

        self.save_dir = save_dir
        self.case_name = next(os.walk(label_folder_dir), (None, None, []))[2]
        self.case_name = self.case_name[0]  # Chỉ lấy một mẫu

        self.MRI_CODE = {"Flair": "0000", "T1w": "0001", "T1gd": "0002", "T2w": "0003", "label": None}

    def normalize(self, x: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_1D_array = scaler.fit_transform(x.reshape(-1, x.shape[-1]))
        normalized_data = normalized_1D_array.reshape(x.shape)
        return normalized_data

    def orient(self, x: MetaTensor) -> MetaTensor:
        assert isinstance(x, MetaTensor)
        return Orientation(axcodes="RAS")(x)

    def detach_meta(self, x: MetaTensor) -> np.ndarray:
        assert isinstance(x, MetaTensor)
        return EnsureType(data_type="numpy", track_meta=False)(x)

    def crop_brats2021_zero_pixels(self, x: np.ndarray) -> np.ndarray:
        return x[:, 56:184, 56:184, 13:141]

    def remove_case_name_artifact(self, case_name: str) -> str:
        return case_name.rsplit(".")[0]

    def get_modality_fp(self, case_name: str, folder: str, mri_code: str = None):
        if mri_code:
            f_name = f"{case_name}_{mri_code}.nii.gz"
        else:
            f_name = f"{case_name}.nii.gz"

        modality_fp = os.path.join(self.train_folder_dir, folder, f_name)
        return modality_fp

    def load_nifti(self, fp):
        nifti_data = nibabel.load(fp)
        nifti_scan = nifti_data.get_fdata()
        affine = nifti_data.affine
        return nifti_scan, affine

    def _2metaTensor(self, nifti_data: np.ndarray, affine_mat: np.ndarray):
        scan = MetaTensor(x=nifti_data, affine=affine_mat)
        D, H, W = scan.shape
        scan = scan.view(1, D, H, W)
        return scan

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

    def __getitem__(self, idx):
        case_name = self.case_name
        case_name = self.remove_case_name_artifact(case_name)

        modalities = []
        for modality in ["Flair", "T1w", "T1gd", "T2w"]:
            code = self.MRI_CODE[modality]
            modality_fp = self.get_modality_fp(case_name, "imagesTr", code)
            mod_data = self.preprocess_brats_modality(modality_fp, is_label=False)
            modalities.append(mod_data.swapaxes(1, 3))  # Transverse plane

        # preprocess segmentation label
        label_fp = self.get_modality_fp(case_name, "labelsTr", self.MRI_CODE["label"])
        label = self.preprocess_brats_modality(label_fp, is_label=True).swapaxes(1, 3)

        # stack modalities (4, D, H, W)
        modalities = np.concatenate(modalities, axis=0, dtype=np.float32)

        return modalities, label, case_name

    def __call__(self):
        print("started preprocessing Brats2017...")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        modalities, label, case_name = self.__getitem__(0)
        data_save_path = os.path.join(self.save_dir, case_name)
        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        modalities_fn = data_save_path + f"/{case_name}_modalities.pt"
        label_fn = data_save_path + f"/{case_name}_label.pt"
        torch.save(modalities, modalities_fn)
        torch.save(label, label_fn)
        print("finished preprocessing Brats2017...")


if __name__ == "__main__":
    brats2017_task1_prep = Brats2017Task1Preprocess(root_dir="./", train_folder_name="train",
                                                    save_dir="../BraTS2017_Training_Data")
    brats2017_task1_prep()
