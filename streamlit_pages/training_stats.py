import streamlit as st
import os
from PIL import Image

def app():
    st.header("Training Stats & Visualizations")
    st.markdown("""
    - **Loss Curves**
    - **Training Progress**
    - **Model Info**
    - **Dataset Stats**
    """)
    # Example: Show loss curve if exists
    loss_curve_path = "outputs/plots/loss_curve.png"
    if os.path.exists(loss_curve_path):
        st.image(Image.open(loss_curve_path), caption="Loss Curve", use_column_width=True)
    else:
        st.info("Loss curve not found. Please check training outputs.")
