import pyvista as pv
import streamlit as st
import nibabel as nib
import time
import matplotlib.pyplot as plt

st.title('3D Visualization')

###########################################################################################
uploaded_file = st.file_uploader("Tải lên file NIfTI (.nii hoặc .nii.gz)", type=["nii", "nii.gz"])

if uploaded_file is not None:
    with open("temp.nii", "wb") as f:
        f.write(uploaded_file.getbuffer())

    img = nib.load("temp.nii")
    data = img.get_fdata()
    original_shape = data.shape

    st.write(f"Kích thước ban đầu: {original_shape}")

    reduction_factor = st.sidebar.slider("Tỷ lệ giảm kích thước (downsampling)", 0.5, 1.2, 1.0, 0.05)
    new_shape = tuple(int(s * reduction_factor) for s in original_shape)
    data_resampled = data  # Giữ nguyên dữ liệu ban đầu cho phần 3D visualization, bỏ qua zoom

    st.write(f"Kích thước sau khi giảm: {data_resampled.shape}")

    colormap = st.sidebar.selectbox("Chọn bảng màu (Colormap)", ["hot", "bone", "viridis", "cool", "magma", "jet"])
    opacity_mode = st.sidebar.selectbox("Kiểu độ trong suốt (Opacity)",
                                        ['sigmoid', 'linear', 'geom', 'geom_r', 'sigmoid_1', 'sigmoid_2', 'sigmoid_3',
                                         'sigmoid_4', 'sigmoid_5', 'sigmoid_6', 'sigmoid_7', 'sigmoid_8', 'sigmoid_9',
                                         'sigmoid_10', 'foreground', 'linear_r', 'sigmoid_r', 'sigmoid_3_r',
                                         'sigmoid_4_r', 'sigmoid_5_r', 'sigmoid_6_r', 'sigmoid_7_r', 'sigmoid_8_r',
                                         'sigmoid_9_r', 'sigmoid_10_r'])
    opacity_unit_distance = st.sidebar.slider("Độ trong suốt theo khoảng cách", 0.1, 5.0, 2.5)
    show_edges = st.sidebar.checkbox("Hiển thị cạnh (Show Edges)", value=False)
    shading = st.sidebar.checkbox("Bật chế độ đổ bóng (Shading)", value=True)
    ambient = st.sidebar.slider("Ánh sáng vùng tối (Ambient)", 0.0, 1.0, 0.2)
    diffuse = st.sidebar.slider("Khuếch tán ánh sáng (Diffuse)", 0.0, 1.0, 0.9)
    specular = st.sidebar.slider("Độ phản chiếu bề mặt (Specular)", 0.0, 1.0, 0.3)
    edge_color = st.sidebar.color_picker("Màu cạnh (Edge Color)", "#ffffff")

    def render_plot():
        grid = pv.UniformGrid()
        grid.dimensions = data_resampled.shape
        grid.spacing = (1, 1, 1)
        grid.point_data["values"] = data_resampled.flatten(order="F")

        plotter = pv.Plotter(window_size=[800, 600])
        plotter.add_volume(
            grid,
            scalars="values",
            cmap=colormap,
            opacity=opacity_mode,
            opacity_unit_distance=opacity_unit_distance,
            shade=shading,
            ambient=ambient,
            diffuse=diffuse,
            specular=specular,
        )

        if show_edges:
            plotter.add_mesh(grid.extract_surface(), color=edge_color, line_width=1)

        plotter.view_isometric()
        plotter.background_color = "white"

        return plotter

    
    def display_mri_slice(fps, cmap, origin, show_grid):
        img = nib.load("temp.nii")
        img_data = img.get_fdata()

        # Chọn chiều dọc theo trục Z để hiển thị các lát cắt
        num_slices = img_data.shape[2]  # Số lát cắt theo chiều Z

        time_per_slice = 1 / fps  # Thời gian cho mỗi lát cắt dựa trên FPS

        # Tạo một container trống để hiển thị các lát cắt
        container = st.empty()

        for slice_index in range(num_slices):  # Duyệt qua tất cả các lát cắt từ 0 đến num_slices-1
            slice_img = img_data[:, :, slice_index]

            # Hiển thị lát cắt ảnh MRI
            fig, ax = plt.subplots()
            ax.imshow(slice_img.T, cmap=cmap, origin=origin)
            ax.grid(show_grid)  # Cập nhật hiển thị lưới dựa trên tham số show_grid
            ax.set_title(f"Lát cắt {slice_index + 1} / {num_slices}")
            
            # Cập nhật hình ảnh trong container
            container.pyplot(fig)

            # Chờ một khoảng thời gian trước khi chuyển sang lát cắt tiếp theo
            time.sleep(time_per_slice)  # Thời gian chuyển tiếp giữa các lát cắt


    # Sử dụng st.session_state để lưu trữ kết quả render_plot
    if 'plotter' not in st.session_state:
        plotter = render_plot()

    tabs = st.tabs(["Tab 1", "Tab 2"])

    with tabs[0]:
        # Hiển thị ảnh 3D trong Tab 1 chỉ khi cần thiết
        if st.button('Render 3D Image'):
            plotter.show()

    with tabs[1]:
       
        fps = st.slider("Khung hình trên giây (FPS)", 1, 60, 10)  # Cho phép người dùng điều chỉnh FPS
        cmap = st.selectbox("Chọn bảng màu (Colormap)", ["hot", "bone", "viridis", "cool", "magma", "jet"], key='select_colormap')
        origin = st.selectbox("Chọn hướng hiển thị ảnh", ['lower', 'upper'])
        show_grid = st.checkbox("Hiển thị lưới (Grid)", value=False)

        if st.button('Show Slide Video'):
            # Hiển thị lát cắt ảnh MRI trong Tab 2 với các tham số đã chọn
            display_mri_slice(fps, cmap, origin, show_grid)

##########################################################################################
