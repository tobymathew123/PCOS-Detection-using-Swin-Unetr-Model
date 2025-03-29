# PCOS Detection Using Swin UNETR

## Overview
This project presents an innovative approach to detecting Polycystic Ovary Syndrome (PCOS) by converting 2D ultrasound images into 3D volumes and analyzing them with MONAI’s Swin UNETR, a state-of-the-art transformer-based model. By slicing each 2D ultrasound image into 32 segments and stacking them into a 3D representation, we enable enhanced visualization and segmentation of ovarian follicles, offering a novel diagnostic tool. Our mission is to improve PCOS detection, particularly in underserved rural areas, while promoting the use of 3D ultrasound imaging for superior clinical insights.

## Key Features
- **3D Transformation**: Converts 2D ultrasound images into 3D volumes (32 slices) for richer data analysis.
- **Model**: Employs MONAI’s Swin UNETR, blending transformer and U-Net architectures for robust 3D segmentation.
- **Novel Approach**: First known use of 2D-to-3D conversion with Swin UNETR for PCOS detection.
- **User Interface**: Streamlit app for intuitive model interaction and visualization.
- **Training**: Executed on Google Colab with T4 GPU support.
- **Impact**: Aims to empower rural healthcare with accessible, AI-driven PCOS diagnostics.

