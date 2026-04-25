import streamlit as st
import torch
from models.discriminator import Discriminator
import config
from PIL import Image
import torchvision.transforms as T

def load_discriminator():
    model = torch.jit.load(config.DISCRIMINATOR_WEIGHTS, map_location="cpu")
    model.eval()
    return model

def predict_real_fake(model, img):
    transform = T.Compose([
        T.Resize((64,64)),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(img_tensor)
        prob = out.item()
    return prob

def app():
    st.header("Real/Fake Anime Face Detector")
    uploaded = st.file_uploader("Upload an anime face image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        model = load_discriminator()
        prob = predict_real_fake(model, img)
        label = "Real" if prob > 0.5 else "Fake"
        st.progress(int(prob*100))
        st.write(f"**Prediction:** {label} ({prob*100:.2f}% confidence)")
