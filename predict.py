import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime
st.set_page_config(page_title="Chicken Disease Detection", layout="wide")

# =====================================================
# CONFIGURATION
# =====================================================
data_dir = r"D:\Accurate Projects\2025-2026\Chicken Disease Image Classification\Processed Dataset"
model_path = os.path.join(data_dir, "EfficientNetB3_best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# TRANSFORM (same as training)
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(pretrained=False)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(1536),
        nn.Linear(1536, 256),
        nn.ReLU(),
        nn.Dropout(0.45),
        nn.Linear(256, 4),
        nn.Softmax(dim=1)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =====================================================
# CLASS LABELS (adjust as per your dataset)
# =====================================================
class_names = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

# =====================================================
# DISEASE INFORMATION DATABASE
# =====================================================
disease_info = {
    "Coccidiosis": {
        "Reason": "Caused by protozoan parasites (Eimeria species) that damage the intestinal lining of chickens, often due to contaminated litter or water.",
        "Prevention": "Keep litter dry, use coccidiostats in feed, maintain strict hygiene, and vaccinate young chicks if available.",
        "Medical Drugs": "Amprolium (Corid), Sulfadimethoxine, Toltrazuril, or Diclazuril as prescribed by a veterinarian.",
        "Recovery Time": "Typically 7–10 days with medication and good supportive care."
    },
    "Healthy": {
        "Reason": "The chicken shows no visible symptoms of infection or abnormality.",
        "Prevention": "Continue balanced diet, maintain clean housing, provide fresh water, and regular vaccination schedule.",
        "Medical Drugs": "No medication required. Maintain vitamin and mineral supplementation for immune support.",
        "Recovery Time": "Not applicable – bird is healthy."
    },
    "NewCastle": {
        "Reason": "Caused by Newcastle disease virus (Paramyxovirus type 1), spread through respiratory secretions or contaminated feed and water.",
        "Prevention": "Vaccinate all poultry regularly, disinfect poultry houses, restrict movement of birds, and isolate new flocks.",
        "Medical Drugs": "No direct antiviral treatment. Supportive therapy using multivitamins (especially A, D, E) and electrolytes.",
        "Recovery Time": "2–3 weeks; mortality can occur if not vaccinated or treated early."
    },
    "Salmonella": {
        "Reason": "Caused by Salmonella bacteria (usually S. pullorum or S. gallinarum), spread through contaminated feed, water, or equipment.",
        "Prevention": "Maintain strict biosecurity, disinfect equipment, provide clean water, and use Salmonella-free chicks and feed.",
        "Medical Drugs": "Enrofloxacin, Neomycin, or Oxytetracycline under veterinary guidance.",
        "Recovery Time": "Typically 10–14 days with antibiotics and supportive therapy."
    }
}

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_image(image_pil):
    img = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return class_names[preds.item()], probs

# =====================================================
# STREAMLIT UI
# =====================================================

st.title(" Chicken Disease Detection using EfficientNetB3")
st.markdown("Upload or capture a chicken image to classify the disease and get prevention & treatment details.")

option = st.radio("Select Input Method", ["Upload Image", "Use Live Camera"])

# -----------------------------------------------------
# UPLOAD IMAGE OPTION
# -----------------------------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a chicken image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width =True)
        st.write("Predicting...")

        label, probs = predict_image(image)
        st.success(f"**Predicted Class:** {label}")

        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        # --- Disease Information ---
        st.subheader("Disease Information")
        info = disease_info[label]
        st.markdown(f"**Reason:** {info['Reason']}")
        st.markdown(f"**Prevention:** {info['Prevention']}")
        st.markdown(f"**Medical Drugs:** {info['Medical Drugs']}")
        st.markdown(f"**Recovery Time:** {info['Recovery Time']}")

# -----------------------------------------------------
# LIVE CAMERA OPTION
# -----------------------------------------------------
elif option == "Use Live Camera":
    st.write("Click the button below to open your webcam.")
    if st.button("Open Camera"):
        cap = cv2.VideoCapture(0)
        st.info("Press 'C' to capture an image, 'Q' to quit.")

        captured_image = None
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            cv2.imshow("Live Feed - Press 'C' to Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') or key == ord('C'):
                captured_image = frame.copy()
                break
            elif key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_image is not None:
            image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            st.image(image_pil, caption="Captured Image", use_container_width =True)

            label, probs = predict_image(image_pil)
            st.success(f"**Predicted Class:** {label}")

            prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            # --- Disease Information ---
            st.subheader("Disease Information")
            info = disease_info[label]
            st.markdown(f"**Reason:** {info['Reason']}")
            st.markdown(f"**Prevention:** {info['Prevention']}")
            st.markdown(f"**Medical Drugs:** {info['Medical Drugs']}")
            st.markdown(f"**Recovery Time:** {info['Recovery Time']}")
