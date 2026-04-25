import streamlit as st

def app():
    st.title("Anime Face Generation with DCGAN")
    st.markdown("""
    Welcome to the Anime Face GAN demo!\n
    **Features:**
    - Generate new anime faces
    - Detect real vs fake anime faces
    - Explore generated images
    - View training stats
    - Learn about GANs and DCGANs
    """)
    st.info("Use the sidebar to navigate between pages.")
