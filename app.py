import streamlit as st
from PIL import Image
import requests
from io import BytesIO

from webscraper import WebScraper

class BirdsongApp:
    def __init__(self):
        self.scraper = WebScraper()

    def run(self):
        # Set page configuration
        st.set_page_config(page_title="Birdsong", layout="centered")

        # Title in bold and subtitle in italics and smaller font, aligned left
        st.markdown("<h1 style='font-weight: bold; text-align: left;'>Birdsong</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-style: italic; font-size: 16px; text-align: left; margin-top: -20px;'>Birdsong recognition</p>", unsafe_allow_html=True)

        # File uploader for audio
        uploaded_file = st.file_uploader(
            "Upload an audio file (MP3 or WAV format)",
            type=["mp3", "wav"]
        )

        if uploaded_file:
            # Mock prediction (static for now)
            predicted_bird = "Blackbird"
            st.audio(uploaded_file, format="audio/wav")

            # Show recognized bird
            st.subheader("🕊️ Recognized Bird:")
            st.markdown(f"**{predicted_bird}**")

            # Search for image using DuckDuckGo
            with st.spinner("🔍 Searching for the bird image..."):
                image_url = self.scraper.get_image_url(predicted_bird)
                if image_url:
                    try:
                        response = requests.get(image_url)
                        response.raise_for_status()
                        image = Image.open(BytesIO(response.content))
                        st.image(image, caption=predicted_bird, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Error loading image: {e}")
                else:
                    st.info("No image found for this bird.")
        else:
            st.info("Please upload an audio file to get started.")
