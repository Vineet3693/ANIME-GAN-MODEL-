import streamlit as st
import torch
from models.generator import Generator
import config
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np

def load_generator():
    model = torch.jit.load(config.GENERATOR_WEIGHTS, map_location="cpu")
    model.eval()
    return model

def generate_face(model, latent_dim=100):
    noise = torch.randn(1, latent_dim, 1, 1)
    with torch.no_grad():
        fake_img = model(noise).detach().cpu()
    img = (fake_img.squeeze().numpy().transpose(1,2,0) + 1) / 2  # [-1,1] to [0,1]
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def app():
    st.header("Generate Anime Faces")
    model = load_generator()
    if st.button("Generate Face"):
        img = generate_face(model)
        st.image(img, caption="Generated Anime Face", use_column_width=True)
        st.success("Face generated!")
