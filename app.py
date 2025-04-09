import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('dermasense_skin_model.h5')


# Disease class names
class_names = ['Acne', 'Melanoma', 'Psoriasis']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 224, 224, 3)

# App UI
st.title("🧬 DermaSense: AI-Powered Skin Disease Detection")
st.write("Upload a skin image and detect whether it’s Acne, Melanoma, or Psoriasis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess & predict
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
