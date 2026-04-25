import streamlit as st
from streamlit_pages import home, generate, detector, gallery, training_stats, about

PAGES = {
    "Home": home,
    "Generate Faces": generate,
    "Real/Fake Detector": detector,
    "Gallery": gallery,
    "Training Stats": training_stats,
    "About": about
}

def main():
    st.set_page_config(page_title="Anime Face GAN", layout="wide")
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.app()

if __name__ == "__main__":
    main()
