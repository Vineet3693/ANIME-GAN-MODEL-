import streamlit as st

def app():
    st.header("About This Project")
    st.markdown("""
    **What is a GAN?**\n
    Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks contest with each other.\n
    **What is DCGAN?**\n
    Deep Convolutional GANs use convolutional layers for image generation and discrimination.\n
    **How does this model work?**\n
    - The Generator creates new anime faces from random noise.\n    - The Discriminator tries to distinguish real anime faces from generated ones.\n
    **Dataset:**\n    - Anime Face Dataset\n
    **Credits:**\n    - Project by VINEET YADAV\n    - Powered by PyTorch, Streamlit\n    """)
