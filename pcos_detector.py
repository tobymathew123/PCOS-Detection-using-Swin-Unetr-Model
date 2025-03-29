
import streamlit as st
import cv2
import numpy as np
import torch
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, ScaleIntensity
from PIL import Image

device = torch.device("cpu")

def load_model():
    model = SwinUNETR(img_size=(32, 128, 128), in_channels=1, out_channels=2, feature_size=48).to(device)
    model.load_state_dict(torch.load(r'C:\Users\tobym\Clg\pyt\pcos_detector_beta_1.pth', map_location=device, weights_only=True))
    model.eval()
    return model

model = load_model()

def preprocess_scan(scan):
    scan = cv2.resize(scan, (128, 128))
    volume = np.stack([scan] * 32, axis=0) 
    volume = np.expand_dims(volume, axis=0)  
    volume = np.expand_dims(volume, axis=1)  
    transforms = Compose([ScaleIntensity()])
    volume = transforms(volume)
    volume = torch.tensor(volume).float().to(device)
    return volume

def predict(volume):
    with torch.no_grad():
        output = model(volume)
        output = output.mean(dim=[2, 3, 4])
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        if probs[0, 0].item() > 0.5:  
            pred = 0
            normal_prob = min(probs[0, 0].item() * 1.3, 0.95)  
            pcos_prob = 1.0 - normal_prob
        else:  
            pred = 1
            pcos_prob = min(probs[0, 0].item() * 1.3, 0.95)  
            normal_prob = 1.0 - pcos_prob
        return pred, pcos_prob * 100, normal_prob * 100


st.set_page_config(page_title="PCOS Detection", page_icon="🩺", layout="wide")
st.markdown("""
    <style>
    .result-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .pcos-detected {
        font-size: 28px;
        color: #FF4081;  
        font-weight: 600;
        text-align: center;
        background: rgba(255, 64, 129, 0.3);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .no-pcos {
        font-size: 28px;
        color: #26A69A;  
        font-weight: 600;
        text-align: center;
        background: rgba(38, 166, 154, 0.3);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)


st.title("PCOS Detection Suite")
st.write("Discover AI-driven diagnostics—upload an ultrasound scan to detect PCOS instantly.")


col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Ultrasound Scan", type=["jpg", "jpeg", "png"], help="Select a scan for analysis.")

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        st.image(image, caption="Your Ultrasound Scan", use_container_width=True)
        scan = np.array(image)
        volume = preprocess_scan(scan)
        pred_class, prob_infected, prob_notinfec = predict(volume)
        
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("Diagnosis Insights")
        if pred_class == 1:
            st.markdown("<div class='pcos-detected'>PCOS Detected!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='no-pcos'>No PCOS Detected</div>", unsafe_allow_html=True)
        st.write(f"**PCOS Probability:** {prob_infected:.2f}%")
        st.write(f"**Normal Probability:** {prob_notinfec:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)