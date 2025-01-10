import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from utils import load_model
from segnet.infer import segresnet_get_predict
from Segformer3d.model import segformer3d_get_predict
import streamlit as st
import zipfile
import numpy as np
import pandas as pd
import math
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix




# Define label mapping
label_mapping = {
    1: "NCR/NET",  # Necrotic and Non-enhancing Tumor Core
    2: "ED",       # Peritumoral Edema
    4: "ET"        # GD-enhancing Tumor
}

# Define model configurations
MODEL_LIST = (
    'dice loss + cross entropy loss, l_rate = 0.0001',
    'dice loss, l_rate = 0.0001',
    'dice loss + cross entropy loss, l_rate = 0.001',
    'dice loss + cross entropy loss, l_rate = 0.01',
    'dice loss + cross entropy loss, l_rate = 0.1'
)
MODEL_DICT = {
    MODEL_LIST[0]: "pretrained1",
    MODEL_LIST[1]: "pretrained2",
    MODEL_LIST[2]: "pretrained3",
    MODEL_LIST[3]: "pretrained4",
    MODEL_LIST[4]: "pretrained5"
}

# Function to calculate segmentation metrics
def calculate_metrics(y_true, y_pred, label_mapping):
    metrics = {}
    for label, label_name in label_mapping.items():
        # Mask for the current label
        true_mask = (y_true == label)
        pred_mask = (y_pred == label)

        # True Positive, False Positive, False Negative
        TP = np.sum(true_mask & pred_mask)
        FP = np.sum(~true_mask & pred_mask)
        FN = np.sum(true_mask & ~pred_mask)

        # Dice Score
        dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

        # Precision, Recall, F1 Score
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Store results
        metrics[label_name] = {
            "Dice Score": dice,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    return metrics



def display_brat_properties():
    # Dữ liệu bảng mô tả các thông số của bệnh nhân trong bộ dữ liệu BRATS 2020
    data = {
        "Property": [
            "Image size (m × n)",
            "Number of MR slices",
            "Slice thickness",
            "Slice separation",
            "Pixel spacing",
            "Magnetic field strength",
            "Number of image type",
            "num class"
        ],
        "Value": [
            "240 × 240 pixels",
            "155 slices",
            "1 mm",
            "1 mm",
            "0.5 mm",
            "3 T",
            "4",
            "4"
        ],
      
    }

    # Tạo DataFrame từ dữ liệu trên
    df = pd.DataFrame(data)

    # Hiển thị bảng
    st.title('Properties of BRATS 2020 Patient MRI Data')
    st.table(df)


# Streamlit application
def app():

    # Configure page settings
    st.set_page_config(page_title="Brain Tumor Segmentation Metrics", layout="wide")

    display_brat_properties()

    # Title and description
    st.title("Brain Tumor Segmentation Metrics Viewer")
    st.write("This app compares models, displays metrics, and visualizes segmentation results.")

    # Display learning curves
    st.subheader("Compare Learning Curves")
    learning_curves_path = "data/data_analysis/accuracy_curves_multiple_loss.png"
    multiple_loss_image = Image.open(learning_curves_path)
    st.image(multiple_loss_image, caption="Learning curves of multiple models.")

    # Display test accuracy
    st.subheader("Test Accuracy Table")
    accuracy_data_path = "data/data_analysis/test_accuracy.csv"
    test_accuracy_df = pd.read_csv(accuracy_data_path)
    st.dataframe(test_accuracy_df, use_container_width=True)

    # Show model-specific details
    st.subheader("Model Details")
    selected_model = st.selectbox("Select a model to view details", MODEL_LIST)
    chosen_model = MODEL_DICT[selected_model]

    # Display model-specific plots
    accuracy_curve_path = f"data/data_analysis/{chosen_model}_accuracy_curve.png"
    train_valid_dices_path = f"data/data_analysis/{chosen_model}_train_valid_dice.png"

    accuracy_curve_image = Image.open(accuracy_curve_path)
    train_valid_dices_image = Image.open(train_valid_dices_path)

    st.image(accuracy_curve_image, caption="Accuracy Curve")
    st.image(train_valid_dices_image, caption="Train vs Validation Dice Scores")




    model_list = ['segresnet_1', 'segresnet_2','segresnet_origin' ,'segformer3d']
    model_name = st.selectbox("Select model", model_list, index=0, key='select model')

    uploaded_zip = st.file_uploader("Upload a zip file containing MRI data", type=["zip"], key='upload input file')


    model = load_model(model_name)

    metrics = None
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


        metrics = calculate_metrics(label_back, pred_back, label_mapping)

        st.header("Segmentation Metrics")
        for label, values in metrics.items():
            st.subheader(f"Metrics for {label}")
            metrics_df = pd.DataFrame([values])
            st.dataframe(metrics_df, use_container_width=True)

        # Summary visualization of metrics
        st.header("Summary Visualization")
        summary_data = {
            "Label": list(metrics.keys()),
            "Precision": [values["Precision"] for values in metrics.values()],
            "Recall": [values["Recall"] for values in metrics.values()],
            "F1 Score": [values["F1 Score"] for values in metrics.values()]
        }

        summary_df = pd.DataFrame(summary_data)

        # Create subplots for each metric (Precision, Recall, F1 Score)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot for Precision
        axes[0].bar(summary_df["Label"], summary_df["Precision"], color='blue')
        axes[0].set_title("Precision")
        axes[0].set_xlabel("Labels")
        axes[0].set_ylabel("Score")

        # Plot for Recall
        axes[1].bar(summary_df["Label"], summary_df["Recall"], color='green')
        axes[1].set_title("Recall")
        axes[1].set_xlabel("Labels")
        axes[1].set_ylabel("Score")

        # Plot for F1 Score
        axes[2].bar(summary_df["Label"], summary_df["F1 Score"], color='red')
        axes[2].set_title("F1 Score")
        axes[2].set_xlabel("Labels")
        axes[2].set_ylabel("Score")

        # Display the plots in Streamlit
        st.header("Summary Visualization")
        st.pyplot(fig)


        label_back = np.array(label_back)  # Dữ liệu nhãn thực tế
        pred_back = np.array(pred_back)    # Dữ liệu nhãn dự đoán

        # Chuyển đổi dữ liệu 3D thành 1D
        label_back_flat = label_back.flatten()
        pred_back_flat = pred_back.flatten()

        # Tính toán confusion matrix sử dụng các giá trị nhãn nguyên (1, 2, 4)
        cm = confusion_matrix(label_back_flat, pred_back_flat, labels=[1, 2, 4])

        # Chuyển nhãn số nguyên sang nhãn chuỗi (ví dụ: "NCR/NET", "ED", "ET") để hiển thị
        cm_labels = ["NCR/NET", "ED", "ET"]

        # Tạo biểu đồ confusion matrix bằng seaborn
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, 
                    yticklabels=cm_labels, ax=ax, cbar=False)

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        # Hiển thị confusion matrix trong Streamlit
        st.pyplot(fig)


# Run the application
if __name__ == "__main__":
    app()
