# Brain Tumor Segmentation Project

## Project Overview
This project involves developing a deep learning-based system for segmenting brain tumors from MRI images. The system is designed to aid in medical diagnostics by accurately identifying and segmenting different regions of brain tumors, leveraging advanced deep learning models and visualization tools.

## Key Features
- **Brain tumor segmentation** using deep learning models.
- **Interactive visualizations** of MRI images in 2D and 3D formats.
- **Statistical analysis** of tumor data to extract medical insights.
- A **user-friendly interface** built with Streamlit for ease of use by researchers and medical professionals.

## Dataset
The project utilizes the **BraTS 2020 dataset**, which contains pre-operative multi-modal MRI scans from glioblastoma (HGG) and lower-grade glioma (LGG) patients. Each image is annotated with ground-truth segmentation for tumor regions.

### Dataset Details
- **MRI Modalities**: T1, T1Gd, T2, FLAIR.
- **Labels**:
  - 0: Background (No tumor).
  - 1: Necrotic and Non-enhancing Tumor Core (NET/NCR).
  - 2: Peritumoral Edema (ED).
  - 3: Enhancing Tumor (ET).
- **File Format**: NIfTI (.nii.gz).
- **Volume Dimensions**: 240 × 240 × 155 voxels.
![Alt text](data/image/brats_dataset.png)
![Alt text](data/image/brats_dataset2.png)



You can read more about dataset in:
[brats 2020](https://www.med.upenn.edu/cbica/brats2020/data.html)\
And the dataset to train model from kaggle: [dataset train](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

## Models Used
### U-Net
- Architecture designed for biomedical image segmentation.
- **Dice Score**: 0.85
- **IoU Score**: 0.76

## Test Results of Pretrained Models
| Model                                         | IOU Score | Dice Score |
|-----------------------------------------------|-----------|------------|
| dice_loss + cross_entropy_loss, l_rate=0.0001 | 0.7465402 | 0.8548798  |
| dice_loss, l_rate=0.0001                      | 0.7388876 | 0.8498393  |
| dice_loss + cross_entropy_loss, l_rate=0.001  | 0.7465402 | 0.8548789  |
| dice_loss + cross_entropy_loss, l_rate=0.01   | 0.7522497 | 0.8586102  |
| dice_loss + cross_entropy_loss, l_rate=0.1    | 0.7797701 | 0.8762593  |

![Alt text](data/image/unet_structure.png)
![Alt text](data/image/predict_single.png)


You can read more about paper of unet in: [Unet paper](https://arxiv.org/pdf/1505.04597)

### SegNet
- Encoder-decoder architecture optimized for semantic segmentation.
- Efficient memory usage and accurate boundary detection.

![Alt text](data/image/segnet_architecture.png)

You can read more about paper of segnet in: [Segnet paper](https://arxiv.org/pdf/1511.00561)

### SegFormer 3D
- Transformer-based architecture tailored for 3D medical imaging.
- Enhanced performance with multi-scale feature learning.

![Alt text](data/image/segformer3d_architecture.png)


You can read more about paper of segformer 3d in: [Segformer 3D paper](https://arxiv.org/pdf/2404.10156)

## Program Features
### Main Functionalities
1. **Model Training**
   - Training on the BraTS 2020 dataset using GPU acceleration.
   - Evaluation metrics: Dice Coefficient, IoU, Precision, Recall.

   This is the many result of training model in this project:
   ![alt text](data/image/result4.png)
   ![alt text](data/image/metric.png)
   ![alt text](data/image/multi_class_result.png)
   ![alt text](data/image/multiclass_result3.png)


   In the program you can infer the model with single image folder and calculate the metric do visualizing as chart:

    ![alt text](data/image/metric_per_instance.png)
    ![alt text](data/image/metric_per_instance2.png)
    ![alt text](data/image/metric_per_instance3.png)
2. **2D and 3D Visualization**
   - Slice-by-slice visualization with adjustable settings.
   - Full 3D reconstruction of tumor regions.

    ![alt text](data/image/3d_param_and_image.png)
    ![alt text](data/image/3d_orther_color.png)
    ![alt text](data/image/3d_orther_mode.png)


3. **Statistical Analysis**
   - Voxel counts and tumor volume calculations.
   - Percentage distribution visualizations (e.g., bar and pie charts).
 ![alt text](data/image/medical_class_percent.png)
 ![alt text](data/image/medical_chart_1.png)
 ![alt text](data/image/medical_chart_2.png)
 ![alt text](data/image/medical_metric.png)
 ![alt text](data/image/medical_position_metric.png)
### Additional Features
- Adjustable visualization settings.
- Flexible input options for model selection and parameter tuning.
- Animated visualizations of MRI slices.

visualizing video:
https://github.com/user-attachments/assets/6846c9e2-d56d-4769-95f7-26b7641ee817



## Installation and Setup
### Prerequisites
- Python 3.8+
- Required Libraries: TensorFlow, PyTorch, Streamlit, NumPy, Matplotlib, etc.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/nhatky160103/Brain-Tumor-segmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Brain-Tumor-segmentation
   ```
2. Create a virtual envirement:
   ```bash
   python -m venv .venv
   ```

2. Active venv:
   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download data and the pretrain model and set all file and folder to the right position:

    You can download data and pretrained model in the following link:
    [data and model](https://drive.google.com/drive/folders/1LN6Ga4gfNvOtDaRi7E_c9AeqyJ3cSH3Z?usp=sharing)

4. Run the Streamlit application:
   ```bash
   streamlit run home.py
   ```

## Usage
1. Upload MRI images in **NIfTI format**.
2. Select the desired model (U-Net, SegNet, or SegFormer 3D).
3. View segmentation results in 2D or 3D.
4. Analyze statistical outputs for detailed insights.


## References
1. Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015). *SegNet: A deep convolutional encoder-decoder architecture for image segmentation*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation*. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).
3. Li, X., et al. (2021). *SegFormer: A segmentation transformer for semantic segmentation*. arXiv preprint arXiv:2105.15203.
4. Brain Tumor Segmentation Challenge (BraTS) 2020 Dataset: [https://www.med.upenn.edu/cbica/brats2020](https://www.med.upenn.edu/cbica/brats2020).

## Future Work
- Enhance model performance with advanced architectures like **3D U-Net**.
- Expand dataset compatibility to include other medical imaging modalities.
- Integrate a more interactive and intuitive user interface.

## Contributors
**Đinh Nhật Ký**  
- Email: [ky.dn215410@sis.hust.edu.vn](mailto:ky.dn215410@sis.hust.edu.vn)  
- GitHub: [nhatky160103](https://github.com/nhatky160103)
