import numpy as np
from scipy.spatial.distance import directed_hausdorff
from utils import load_model
from segnet.infer import segresnet_get_predict
from Segformer3d.model import segformer3d_get_predict
import streamlit as st
import torch
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import altair as alt
import plotly.graph_objects as go
from scipy.ndimage import label, center_of_mass
from skimage.measure import label, regionprops
import math
from skimage.measure import label, regionprops
from skimage import measure
from skimage.measure import marching_cubes
from scipy.spatial import ConvexHull
import math
import plotly.graph_objects as go




import os
st.title("Medical metric ")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_volume(label, voxel_volume=1.0):
    # Dictionary ánh xạ nhãn số sang tên
    label_mapping = {
        1: "NCR/NET",  # Necrotic and Non-enhancing Tumor Core
        2: "ED",       # Peritumoral Edema
        4: "ET"        # GD-enhancing Tumor
    }

    # Tính toán số lượng voxel cho mỗi nhãn
    unique_labels, counts = np.unique(label, return_counts=True)
    total_volume = np.sum(counts) * voxel_volume

    # Tạo một dictionary chứa volume và phần trăm cho mỗi nhãn
    volumes = {
        label_mapping.get(int(lbl), f"Unknown ({lbl})"): {  # Sử dụng tên thay vì số
            "volume": cnt * voxel_volume,
            "percentage": (cnt * voxel_volume / total_volume) * 100
        } for lbl, cnt in zip(unique_labels, counts)
    }

    # Chuyển thông tin thành DataFrame để trực quan hóa
    data = pd.DataFrame({
        "Label": list(volumes.keys()),
        "Voxel Count": [volumes[lbl]["volume"] for lbl in volumes],
        "Percentage (%)": [volumes[lbl]["percentage"] for lbl in volumes]
    })

    # Hiển thị bảng dữ liệu
    st.title("Voxel Analysis by Labels")
    st.subheader("Data Table")
    st.dataframe(data)

    # Biểu đồ tương tác với Plotly: Biểu đồ cột (Bar chart)
    st.subheader("Interactive Bar Chart: Voxel Counts per Label (Plotly)")
    fig_bar = px.bar(data, x="Label", y="Voxel Count", text="Voxel Count", title="Voxel Counts per Label")
    fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig_bar)

    # Biểu đồ tương tác với Plotly: Biểu đồ tròn (Pie chart)
    st.subheader("Interactive Pie Chart: Percentage Distribution (Plotly)")
    fig_pie = px.pie(data, names="Label", values="Percentage (%)", title="Percentage Distribution by Label", hole=0.3)
    st.plotly_chart(fig_pie)

    # Biểu đồ tương tác với Altair: Biểu đồ đường (Line chart)
    st.subheader("Interactive Line Chart: Voxel Count (Altair)")
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("Label:O", title="Label"),
        y=alt.Y("Voxel Count:Q", title="Voxel Count"),
        tooltip=["Label", "Voxel Count", "Percentage (%)"]
    ).properties(
        title="Voxel Count by Label"
    )
    st.altair_chart(chart)


    return volumes


def provide_medical_insights(volumes):
    """
    Đưa ra đánh giá và lời khuyên y tế dựa trên thể tích các vùng khối u.

    Các chỉ số phân tích bao gồm thể tích từng vùng (NCR/NET, ED, ET) và đưa ra lời khuyên cụ thể.

    Parameters:
    volumes (dict): Thông tin về thể tích và phần trăm của từng vùng khối u.

    Returns:
    insights (str): Đánh giá và lời khuyên cho bác sĩ.
    """
    insights_net = []
    insights_et = []
    insights_ed = []
    insights_total = []

    # Kiểm tra thể tích các vùng khối u
    ncr_net_volume = volumes.get("NCR/NET", {}).get("volume", 0)
    ed_volume = volumes.get("ED", {}).get("volume", 0)
    et_volume = volumes.get("ET", {}).get("volume", 0)

    # Tính tổng thể tích
    total_volume = ncr_net_volume + ed_volume + et_volume

    # Phân tích từng vùng khối u

    # 1. Vùng NCR/NET (hoại tử và khối u không tăng cường)

    if ncr_net_volume > 0.4 * total_volume:
        insights_net.append(
            "Đây là một dấu hiệu cho thấy khối u có thể đã lan rộng ra các mô xung quanh, có thể ảnh hưởng đến chức năng của các cấu trúc lân cận.")
        insights_net.append(
            "Khuyến nghị: Cần thực hiện đánh giá chi tiết bằng sinh thiết để xác định mức độ ác tính của khối u, cũng như lên kế hoạch điều trị (phẫu thuật hoặc hóa trị).")
    elif ncr_net_volume == 0:
        insights_net.append(
            "Không phát hiện vùng NCR/NET, điều này có thể là dấu hiệu tốt hoặc cần kiểm tra lại dữ liệu. Khối u có thể chưa tiến triển.")
    else:
        insights_net.append(
            "Khối u có thể đang ở giai đoạn không quá nghiêm trọng, nhưng cần theo dõi sát sao. Khuyến nghị kiểm tra lại sau một khoảng thời gian.")



    # 2. Vùng ED (phù nề quanh khối u)
    insights_ed.append(
        f"\n Vùng ED (Peritumoral Edema) chiếm thể tích: {ed_volume:.2f} voxel ({(ed_volume / total_volume) * 100:.2f}% của tổng thể tích).")

    if ed_volume > 0.3 * total_volume:
        insights_ed.append(
            "\n Vùng ED lớn cho thấy có sự phản ứng viêm hoặc áp lực từ khối u lên các mô xung quanh, có thể gây tổn thương và làm suy giảm chức năng vùng lân cận.")
        insights_ed.append(
            "Khuyến nghị: Cân nhắc sử dụng corticosteroid hoặc các phương pháp điều trị giảm viêm để giảm bớt sự phù nề và cải thiện tình trạng mô xung quanh.")
    elif ed_volume == 0:
        insights_ed.append(
            "Không phát hiện vùng ED, điều này có thể là dấu hiệu tốt hoặc khối u chưa gây phản ứng viêm. Cần theo dõi thêm.")
    else:
        insights_ed.append(
            "Vùng ED vừa phải, có thể cho thấy một mức độ phù nề vừa phải. Theo dõi và đánh giá định kỳ để xác định hướng điều trị phù hợp.")

    # 3. Vùng ET (tăng cường tín hiệu)

    if et_volume > 0.3 * total_volume:
        insights_et.append(
            "Vùng ET lớn, cho thấy khối u đang phát triển mạnh và có khả năng xâm lấn cao. Khối u có thể gia tăng khả năng di căn."
        )
        insights_et.append(
            "Khuyến nghị: Ưu tiên các phương pháp điều trị tấn công, chẳng hạn như xạ trị hoặc hóa trị liều cao, nhằm làm giảm kích thước khối u và ngăn ngừa di căn.")
    elif et_volume < 0.1 * total_volume:
        insights_et.append(
            "Vùng ET nhỏ, có thể khối u ít hoạt động hoặc đang trong giai đoạn ổn định. Tuy nhiên, cần theo dõi định kỳ để đảm bảo tình trạng không thay đổi.")
    else:
        insights_et.append(
            "Vùng ET có kích thước trung bình, có thể chỉ ra khối u đang phát triển chậm hoặc đang ở giai đoạn ổn định. Cần tiếp tục theo dõi và đánh giá để quyết định phương pháp điều trị phù hợp.")

    # Tổng quan thể tích khối u

    insights_total.append(f"Tổng thể tích khối u (NCR/NET + ED + ET): {total_volume:.2f} voxel.")

    if total_volume > 100000:  # Giả sử 100000 voxel là mức thể tích lớn
        insights_total.append(
            "Thể tích tổng thể lớn, đây là một tình trạng khối u phức tạp và có thể yêu cầu phẫu thuật hoặc phương pháp điều trị mạnh mẽ.")
        insights_total.append(
            "Khuyến nghị: Cần đánh giá tổng thể tình trạng sức khỏe của bệnh nhân và quyết định phương pháp điều trị tối ưu.")
    elif total_volume < 10000:  # Giả sử 10000 voxel là mức thể tích nhỏ
        insights_total.append(
            "Thể tích khối u nhỏ, có thể là dấu hiệu sớm hoặc giai đoạn ổn định, nên theo dõi định kỳ và kiểm tra sau một khoảng thời gian.")


    # Tóm tắt lời khuyên
    insights_total.append("\nLời khuyên tổng quát:")
    insights_total.append("1. Cần theo dõi định kỳ các thay đổi trong thể tích các vùng khối u.")
    insights_total.append("2. Nếu thể tích tăng nhanh, cần thực hiện các biện pháp điều trị kịp thời.")
    insights_total.append(
        "3. Khuyến khích bác sĩ phối hợp với các chuyên gia khác để đưa ra phương án điều trị tối ưu cho bệnh nhân.")

    return "\n".join(insights_net) ,  "\n".join(insights_ed) , "\n".join(insights_et), "\n".join(insights_total)



def analyze_tumor_shape(prediction, brain_shape=(240, 240, 155)):
    """
    Phân tích hình dạng và vị trí khối u, đánh giá khả năng xâm lấn vào các vùng khác của não
    Dựa trên dữ liệu đầu vào của một phân đoạn dự đoán khối u trong bộ não.

    Parameters:
    - prediction: mảng 3D chứa các giá trị phân đoạn dự đoán khối u
    - brain_shape: kích thước của não (240, 240, 155) cho bộ dữ liệu BRATS
    """

    # 1. Tìm các voxel của khối u
    tumor_voxels = prediction > 0
    tumor_points = np.argwhere(tumor_voxels)
    if len(tumor_points) == 0:
        st.write("Không tìm thấy khối u trong hình ảnh.")
        return

    # 2. Đánh giá vị trí khối u (trong các trục: X, Y, Z)
    center_of_tumor = center_of_mass(tumor_voxels)
    st.write(f"Trung tâm khối u: {center_of_tumor}")

    # Xác định vị trí tương đối của khối u trong bộ não (vùng nào của não)
    brain_center = np.array(brain_shape) / 2
    relative_position = np.array(center_of_tumor) - brain_center
    st.write(f"Khối u lệch {relative_position[0]:.2f} voxel theo trục X, "
             f"{relative_position[1]:.2f} voxel theo trục Y, "
             f"{relative_position[2]:.2f} voxel theo trục Z.")

    # 3. Đánh giá kích thước và hình dạng khối u
    tumor_extent = np.ptp(tumor_points, axis=0)
    st.write(f"Khối u chiếm kích thước: X: {tumor_extent[0]}, Y: {tumor_extent[1]}, Z: {tumor_extent[2]}")

    # 4. Trực quan hóa 3D
    tumor_points = np.array(tumor_points)  # Các điểm của khối u
    x, y, z = tumor_points[:, 0], tumor_points[:, 1], tumor_points[:, 2]  # Các tọa độ voxel của khối u

    # Tạo trực quan hóa 3D với các tùy chọn tùy chỉnh
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,  # Tăng kích thước marker
            color=z,  # Màu sắc marker dựa trên giá trị Z
            colorscale='Jet',  # Dùng colorscale 'Jet' (hoặc 'Hot' tùy thích)
            opacity=0.6,  # Điều chỉnh độ mờ của marker
            line=dict(width=0)  # Loại bỏ viền của marker
        )
    )])

    # Cập nhật layout để trực quan hóa đẹp hơn
    fig.update_layout(
        title="Khối u trong não (Trực quan hóa 3D)",
        scene=dict(
            xaxis_title='Trục X',
            yaxis_title='Trục Y',
            zaxis_title='Trục Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Đặt vị trí camera để có cái nhìn tốt hơn
            ),
            aspectmode="cube"  # Cân bằng tỷ lệ các trục X, Y, Z
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),  # Tùy chỉnh margin
        showlegend=False  # Ẩn legend nếu không cần thiết
    )

    st.plotly_chart(fig)

    # 5. Visualize khối u với các phân đoạn chính
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, axis in enumerate(range(3)):
        # Trực quan hóa mặt cắt trên các trục X, Y, Z
        if axis == 0:
            slice_img = np.sum(prediction, axis=0)
        elif axis == 1:
            slice_img = np.sum(prediction, axis=1)
        else:
            slice_img = np.sum(prediction, axis=2)

        ax[i].imshow(slice_img, cmap='hot', interpolation='nearest')
        ax[i].set_title(f'Projection along axis {axis}')
        ax[i].axis('off')

    # Chuyển từ matplotlib figure sang streamlit plot
    st.pyplot(fig)

    # 6. Đưa ra đánh giá y tế
    evaluation = ""
    if relative_position[0] > 0:
        evaluation += "Khối u lệch sang phải, "
    else:
        evaluation += "Khối u lệch sang trái, "

    if relative_position[1] > 0:
        evaluation += "khối u nằm trên não, "
    else:
        evaluation += "khối u nằm dưới não, "

    if relative_position[2] > 0:
        evaluation += "và khối u gần với vỏ não."
    else:
        evaluation += "và khối u gần với trung tâm não."

    st.write("\nĐánh giá hình dạng khối u:")
    st.write(evaluation)
    st.write(f"Khối u có kích thước {tumor_extent[0]} x {tumor_extent[1]} x {tumor_extent[2]} voxel.")

    # 7. Đưa ra lời khuyên cho bác sĩ
    advice = "Bác sĩ nên xem xét việc xác định khối u theo các hướng sau:\n"
    if tumor_extent[0] > 50:
        advice += "- Khối u có chiều dài lớn, có thể cần chụp thêm MRI để xác định mức độ xâm lấn.\n"
    if tumor_extent[1] > 50:
        advice += "- Khối u có chiều rộng lớn, cần xem xét nguy cơ xâm lấn vào các vùng chức năng của não.\n"
    if tumor_extent[2] > 50:
        advice += "- Khối u có chiều sâu lớn, cần quan tâm đến các cấu trúc quan trọng ở phía sâu não.\n"

    st.write(advice)




def compute_tumor_metrics_and_plot(tensor, tumor_labels=[1, 2, 4], voxel_volume=1):
    """
    Tính toán các thông số của khối u và hiển thị đánh giá, nhận xét y học trên Streamlit.
    """
    # Mapping cho nhãn
    label_mapping = {
        1: "NCR/NET (Necrotic and Non-enhancing Tumor Core)",
        2: "ED (Peritumoral Edema)",
        4: "ET (GD-enhancing Tumor)"
    }
    
    # 1. Tính toán thông số
    tumor_volumes = {}
    for label in tumor_labels:
        label_voxels = (tensor == label)
        tumor_volumes[label] = np.sum(label_voxels) * voxel_volume

    tumor_voxels = np.isin(tensor, tumor_labels)
    total_tumor_volume = np.sum(tumor_voxels) * voxel_volume

    verts, faces, _, _ = marching_cubes(tumor_voxels, level=0.5)
    hull = ConvexHull(verts)
    surface_area = hull.area
    sphericity = (36 * math.pi * total_tumor_volume ** 2) / (surface_area ** 3)
    dims = np.array([np.max(verts[:, i]) - np.min(verts[:, i]) for i in range(3)])
    asymmetry = np.max(dims) / np.min(dims)
    centroid = np.mean(verts, axis=0)
    distances = np.linalg.norm(verts - centroid, axis=1)
    dispersion = np.std(distances)
    tumor_density = total_tumor_volume / surface_area
    
    # 2. Hiển thị đồ thị 3D
    st.subheader("Tumor Visualization")
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.5,
        color='blue'
    )])
    fig.update_layout(scene=dict(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        zaxis=dict(showgrid=False),
        aspectmode="cube"),
        title="Tumor Surface Visualization")
    st.plotly_chart(fig)
    
    # 3. Hiển thị các thông số
    st.subheader("Tumor Metrics")
    st.write(f"**Total Tumor Volume**: {total_tumor_volume:.2f} mm³")
    st.write(f"**Surface Area**: {surface_area:.2f} mm²")
    st.write(f"**Sphericity**: {sphericity:.2f}")
    st.write(f"**Asymmetry Ratio**: {asymmetry:.2f}")
    st.write(f"**Dispersion**: {dispersion:.2f} mm")
    st.write(f"**Tumor Density**: {tumor_density:.2f} mm³/mm²")
    st.write("**Individual Tumor Volumes**:")
    for label, volume in tumor_volumes.items():
        st.write(f"  {label_mapping.get(label, f'Label {label}')}: {volume:.2f} mm³")

    # 4. Đánh giá y học
    st.subheader("Medical Analysis & Recommendations")
    if tumor_density > 5:
        st.write("🔵 **Tumor Density**: Mật độ cao cho thấy khối u có khả năng phát triển đặc và khó can thiệp.")
    else:
        st.write("🟢 **Tumor Density**: Mật độ thấp, điều này có thể thuận lợi hơn cho việc điều trị.")

    if asymmetry > 1.5:
        st.write("🔴 **Asymmetry Ratio**: Khối u có hình dạng bất đối xứng rõ rệt, có thể là dấu hiệu của khối u ác tính hoặc phát triển không đồng đều.")
    else:
        st.write("🟢 **Asymmetry Ratio**: Hình dạng khá đối xứng, đây là dấu hiệu tích cực.")

    if sphericity < 0.8:
        st.write("🔴 **Sphericity**: Khối u có hình dạng không tròn, có thể gợi ý tính chất xâm lấn cao.")
    else:
        st.write("🟢 **Sphericity**: Hình dạng tròn, điều này thường liên quan đến khối u lành tính hơn.")

    if dispersion > 10:
        st.write("🔴 **Dispersion**: Độ phân tán cao cho thấy các phần của khối u không đồng đều, có thể làm phức tạp việc phẫu thuật.")
    else:
        st.write("🟢 **Dispersion**: Độ phân tán thấp, khối u tập trung gần tâm, thuận lợi cho việc xử lý.")

    if surface_area > 1000:
        st.write("🔴 **Surface Area**: Diện tích bề mặt lớn, khối u có thể ảnh hưởng đến nhiều vùng lân cận.")
    else:
        st.write("🟢 **Surface Area**: Diện tích bề mặt nhỏ, khả năng ảnh hưởng hạn chế.")

    # Gợi ý hành động
    st.markdown("---")
    st.subheader("General Recommendations")
    st.write("✔️ Tham khảo bác sĩ chuyên môn để xác định liệu pháp điều trị phù hợp (phẫu thuật, hóa trị, hoặc xạ trị).")
    st.write("✔️ Xem xét so sánh với dữ liệu trước đây (nếu có) để đánh giá sự tiến triển của khối u.")
    st.write("✔️ Kết hợp các xét nghiệm hình ảnh khác như MRI hoặc CT để có cái nhìn tổng quát hơn.")






model_list = ['segresnet_1', 'segresnet_2','segresnet_origin' ,'segformer3d']
model_name = st.selectbox("Select model", model_list, index=0, key='select model')

uploaded_zip = st.file_uploader("Upload a zip file containing MRI data", type=["zip"], key='upload input file')


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

    print(image.shape)
    print(pred.shape)
    print(pred_back.shape)

    import numpy as np

    def count_non_zero_voxels(image):
        non_zero_voxels = np.count_nonzero(image)
        
        return non_zero_voxels

    non_zero_voxels = count_non_zero_voxels(image)
    print(f"Số lượng voxel khác không: {non_zero_voxels}")


    tabs = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
    with tabs[0]:

        volumes = calculate_volume(pred_back, voxel_volume=1.0)
        print("Khối lượng (mm³) của các vùng:", volumes)

        st.subheader("Medical Insights")
        insights_net, insights_ed, insights_et, insights_total = provide_medical_insights(volumes)

        st.markdown("---")
        st.write(insights_net, unsafe_allow_html=True)
        st.markdown("---")
        st.write(insights_ed, unsafe_allow_html=True)
        st.markdown("---")
        st.write(insights_et, unsafe_allow_html=True)
        st.markdown("---")
        st.write(insights_total, unsafe_allow_html=True)
        st.markdown("---")
  
    with tabs[1]:
        metrics = compute_tumor_metrics_and_plot(pred_back)

    with tabs[2]:
        analyze_tumor_shape(pred_back)




