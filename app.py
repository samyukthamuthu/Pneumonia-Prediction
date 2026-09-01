import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# --------------------------------------------------
# Model Path
# --------------------------------------------------

MODEL_PATH = "pneumonia_efficientnetb0.keras"

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("🫁 Pneumonia Detection from Chest X-Ray")

st.write(
    "Upload a chest X-ray image to classify it as "
    "**NORMAL** or **PNEUMONIA**."
)

st.info(
    "This application is intended for educational and research purposes "
    "and should not be used as a substitute for professional medical diagnosis."
)

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Chest X-Ray",
        use_container_width=True
    )

    # Resize image
    image_resized = image.resize((224, 224))

    # Convert image to array
    image_array = np.array(image_resized)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array, verbose=0)[0][0]

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    if prediction >= 0.5:

        predicted_class = "PNEUMONIA"
        confidence = prediction * 100

    else:

        predicted_class = "NORMAL"
        confidence = (1 - prediction) * 100

    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.subheader("Prediction Result")

    if predicted_class == "PNEUMONIA":

        st.error(
            f"Prediction: PNEUMONIA\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"Prediction: NORMAL\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    # --------------------------------------------------
    # Probability Information
    # --------------------------------------------------

    st.subheader("Prediction Probabilities")

    pneumonia_probability = prediction * 100
    normal_probability = (1 - prediction) * 100

    st.write(f"**NORMAL:** {normal_probability:.2f}%")
    st.progress(float(normal_probability / 100))

    st.write(f"**PNEUMONIA:** {pneumonia_probability:.2f}%")
    st.progress(float(pneumonia_probability / 100))