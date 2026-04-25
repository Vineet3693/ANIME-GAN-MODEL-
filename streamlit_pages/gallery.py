import streamlit as st
import os
from PIL import Image
import config

def app():
    st.header("Gallery of Generated Faces")
    img_dir = config.GENERATED_IMAGES_DIR
    if not os.path.exists(img_dir):
        st.warning("No generated images found.")
        return
    files = [f for f in os.listdir(img_dir) if f.endswith((".png",".jpg",".jpeg"))]
    if not files:
        st.warning("No images to display.")
        return
    cols = st.columns(4)
    for i, file in enumerate(files):
        img = Image.open(os.path.join(img_dir, file))
        with cols[i%4]:
            st.image(img, caption=file, use_column_width=True)
            st.download_button("Download", data=img.tobytes(), file_name=file)
