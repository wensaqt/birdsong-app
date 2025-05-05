import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
import librosa
import numpy as np

from webscraper import WebScraper

BIRD_CLASS_MAP = {
    "wlwwar": "Willow Warbler",
    "woosan": "Cape Robin-Chat",
    "comsan": "White-throated Robin-chat",
    "barswa": "Barn Swallow",
    "eaywag1": "Mountain Wagtail",
    "thrni1": "Black-throated Thrush",
    "fatrav1": "Village Weaver",
    "cabgre1": "Great Crested Grebe",
    "abethr1": "Abyssinian Thrush",
    "bagwea1": "Baglafecht Weaver",
    "darbar1": "Northern Masked Weaver",
    "easmog1": "Northern Grey-headed Sparrow"
}

class BirdsongApp:
    def __init__(self):
        self.scraper = WebScraper()
        self.model = tf.keras.models.load_model("models/bird_audio_classifier_EfficientNetB0.h5")

    def preprocess_audio(self, file):
        # Charge l'audio avec librosa pour supporter les formats .ogg
        audio_data, sr = librosa.load(file, sr=None)  # sr=None pour garder la fréquence d'échantillonnage d'origine

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
        log_mel_spec = librosa.util.fix_length(log_mel_spec, size=128, axis=1)
        log_mel_spec = log_mel_spec[..., np.newaxis]  # (128, 128, 1)

        return np.expand_dims(log_mel_spec, axis=0)   # (1, 128, 128, 1)

    def decode_prediction(self, prediction):
        predicted_index = np.argmax(prediction)
        class_keys = list(BIRD_CLASS_MAP.keys())
        predicted_label = class_keys[predicted_index]
        
        print(f"Predicted class key: {predicted_label}")
        print(f"Predicted index: {predicted_index}")
        print(f"Prediction vector: {prediction}")

        return BIRD_CLASS_MAP.get(predicted_label, "Unknown Bird")

    def run(self):
        st.set_page_config(page_title="Birdsong", layout="centered")

        st.markdown("<h1 style='font-weight: bold; text-align: left;'>Birdsong</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-style: italic; font-size: 16px; text-align: left; margin-top: -20px;'>Birdsong recognition</p>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV, or OGG format)", type=["mp3", "wav", "ogg"])

        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")

            with st.spinner("🧠 Classifying bird song..."):
                input_tensor = self.preprocess_audio(uploaded_file)
                prediction = self.model.predict(input_tensor)
                predicted_bird = self.decode_prediction(prediction)

            st.subheader("🕊️ Recognized Bird:")
            st.markdown(f"**{predicted_bird}**")

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
