import json
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# ── Paths ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/vehicle_type_resnet18_v2.pth"
CLASS_PATH = "models/class_names_v2.json"

# ── Page config (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="Smart Valet Vehicle Classifier",
    page_icon="🚗",
    layout="centered"
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }
        .tip-box {
            background: #1e1e2e;
            border-left: 4px solid #f0a500;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.88rem;
            color: #ccc;
            margin-bottom: 1.2rem;
        }
        .result-card {
            background: #1e1e2e;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


# ── Model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    model = models.resnet18(weights=None)
    num_features = model.fc.in_features

    # Must exactly match the head used during training
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, len(class_names))
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    )
    model.eval()
    return model, class_names


# ── Transform (same as val_transform in training) ───────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ── Prediction ──────────────────────────────────────────────────────────
def predict_vehicle(image, model, class_names):
    image        = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs       = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, 1)

    vehicle_type     = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100
    return vehicle_type, confidence_score, probabilities[0]


# ── Parking logic ────────────────────────────────────────────────────────
def assign_parking(vehicle_type):
    parking_rules = {
        "Hatchback": ("Compact Parking Zone",  "🟦"),
        "Sedan":     ("Standard Parking Zone", "🟩"),
        "SUV":       ("Large Vehicle Zone",    "🟧"),
        "Pickup":    ("Pickup / Truck Zone",   "🟥"),
        "Other":     ("Manual Review Zone",    "⬜"),
    }
    zone, icon = parking_rules.get(vehicle_type, ("Manual Review Zone", "⬜"))
    return zone, icon


# ── UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚗 Smart Valet Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by ResNet18 · 5 vehicle classes</div>', unsafe_allow_html=True)

model, class_names = load_model()

# Tip banner
st.markdown("""
<div class="tip-box">
    📷 <strong>Best results:</strong> Photograph the vehicle from the <strong>side</strong>,
    in good natural light, with the full vehicle in frame.
    Front/rear angles may reduce accuracy.
</div>
""", unsafe_allow_html=True)

# Input method
input_method = st.radio(
    "Choose input method:",
    ["📁 Upload Image", "📷 Take Photo with Camera"],
    horizontal=True
)

source = None

if input_method == "📁 Upload Image":
    source = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
else:
    st.info("On mobile: tap **Take Photo** below to open your camera directly.")
    source = st.camera_input("Point your camera at the vehicle and tap capture")

# ── Results ──────────────────────────────────────────────────────────────
if source is not None:
    image = Image.open(source)

    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Classifying vehicle..."):
        vehicle_type, confidence, probabilities = predict_vehicle(image, model, class_names)
        parking_zone, zone_icon = assign_parking(vehicle_type)

    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vehicle Type", vehicle_type)
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")

    # Confidence indicator
    if confidence >= 85:
        st.success("✅ High confidence prediction")
    elif confidence >= 60:
        st.warning("⚠️ Moderate confidence — try a clearer side-angle photo")
    else:
        st.error("❌ Low confidence — the model is uncertain. Try a different angle or lighting")

    st.divider()
    st.subheader("🅿️ Parking Assignment")
    st.markdown(f"### {zone_icon} {parking_zone}")

    st.divider()
    st.subheader("📈 Confidence Across All Classes")
    for cls, prob in zip(class_names, probabilities):
        prob_val = float(prob)
        highlight = "**" if cls == vehicle_type else ""
        st.progress(prob_val, text=f"{highlight}{cls}{highlight}: {prob_val*100:.1f}%")

    st.divider()
    st.caption(
        "Model: ResNet18 fine-tuned · Classes: Hatchback, Other, Pickup, Sedan, SUV · "
        "For best accuracy photograph vehicles from the side in clear daylight."
    )
