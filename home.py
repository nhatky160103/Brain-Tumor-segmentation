import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Brain Tumor Segmentation App", layout="wide")
st.title("Brain Tumor Segmentation Project")

st.write("Welcome to the Brain Tumor Segmentation App!")

# Overview Section
st.header("Overview")
st.write("This app is designed to help visualize and analyze brain tumor MRI scans. It offers an intuitive interface for researchersand  doctors to explore and gain insights into brain tumor data.")
st.image("data/data_analysis/mri-scan.jpg", caption="Illustration of a Brain Tumor MRI Scan", use_container_width =True)

st.markdown("""
**Key Features:**
- Interactive visualization of MRI slices.
- 3D modeling of brain tumors.(single class and mutil class)
- Statistical insights and recommendations.
""")

# Visualize Slices Section
st.header("Visualize Slices")
st.write("Explore detailed MRI slices of the brain to locate and analyze tumors along slice views.")
st.image("data/data_analysis/slice_visual.png", caption="Example of MRI Slice Visualization", use_container_width =True)
st.markdown("""
**Why it Matters:**
- Provides a clear understanding of the tumor's position.
- Allows for change many color mode and vistalization.
""")

# 3D Tumor Modeling Section
st.header("3D Tumor Modeling")
st.write("Create a segmentation model to perform tumor recognition and visualization")
st.write("Visualize the tumor in three dimensions to better understand its structure, size, and shape. This helps in planning surgical approaches or monitoring treatment efficacy.")
st.image("data/data_analysis/prediction page.png", caption="Sample 2D Visualization of a Tumor single class", use_container_width =True)
st.image("data/data_analysis/brats 2020 data.png", caption="Sample 2D Visualization of a Tumor multi class", use_container_width =True)
st.image("data/data_analysis/3d tumor.png", caption="Sample 3D Visualization of a Tumor", use_container_width =True)
st.markdown("""
**Key Benefits:**
- Provides a holistic view of the tumor.
- Enhances the ability to find the tumor and its size and its sub part.
""")

# Tumor Statistics Section
st.header("Tumor Statistics")
st.write("Analyze the tumor data to extract meaningful statistics, such as tumor volume, shape complexity, position, volume, volume percent of each class in the tumor. These insights can guide diagnosis and treatment decisions.")
st.image("data/data_analysis/mri_static.png", caption="Example of Tumor Statistical Analysis", use_container_width =True)
st.markdown("""
**Insights Provided:**
- Quantitative measurements of the tumor's dimensions and volume.
- Recommendations based on the tumor's characteristics.
""")

# Sidebar Section
st.sidebar.markdown("---")
st.sidebar.title("Main function")
st.sidebar.markdown("""
- Overview
- Visualize Slices
- 3D Tumor Modeling
- Tumor Statistics
""")
st.sidebar.markdown("---")
st.sidebar.write("Created by Dinh Nhat Ky 20215410")
