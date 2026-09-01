import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageStat

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = "pneumonia_efficientnetb0.keras"


@st.cache_resource
def load_pneumonia_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_pneumonia_model()

# --------------------------------------------------
# Image Validation
# --------------------------------------------------

def is_likely_chest_xray(image):
    """
    Basic validation to reject obviously non-X-ray images.

    This is a screening check, not a medical image validator.
    """

    # Convert to RGB for consistent processing
    rgb_image = image.convert("RGB")

    # Convert to grayscale
    gray = rgb_image.convert("L")

    # Resize for analysis
    gray = gray.resize((224, 224))

    # Convert to numpy
    img_array = np.array(gray).astype(np.float32)

    # ----------------------------------------------
    # Check 1: Image should have meaningful intensity
    # ----------------------------------------------

    mean_intensity = np.mean(img_array)
    std_intensity = np.std(img_array)

    if std_intensity < 20:
        return False

    # ----------------------------------------------
    # Check 2: Chest X-rays are usually grayscale
    # ----------------------------------------------

    rgb_array = np.array(rgb_image).astype(np.float32)

    channel_difference = np.mean(
        np.abs(rgb_array[:, :, 0] - rgb_array[:, :, 1])
    ) + np.mean(
        np.abs(rgb_array[:, :, 1] - rgb_array[:, :, 2])
    )

    # Strongly colored images are likely not X-rays
    if channel_difference > 80:
        return False

    # ----------------------------------------------
    # Check 3: Avoid very small images
    # ----------------------------------------------

    width, height = image.size

    if width < 100 or height < 100:
        return False

    # ----------------------------------------------
    # Basic X-ray intensity range
    # ----------------------------------------------

    if mean_intensity < 25 or mean_intensity > 235:
        return False

    return True


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🫁 Pneumonia Detection from Chest X-Ray")

st.write(
    "Upload a chest X-ray image to classify the image as "
    "**NORMAL** or **PNEUMONIA**."
)

# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)

        # ------------------------------------------
        # Display uploaded image
        # ------------------------------------------

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # ------------------------------------------
        # Validate image
        # ------------------------------------------

        if not is_likely_chest_xray(image):

            st.error(
                "❌ Invalid Image: Please upload a valid chest X-ray image."
            )

            st.stop()

        # ------------------------------------------
        # Valid image
        # ------------------------------------------

        st.success("✅ Chest X-ray image detected.")

        # Convert RGB
        image = image.convert("RGB")

        # Resize
        image_resized = image.resize((224, 224))

        # Convert to NumPy
        image_array = np.array(
            image_resized,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )

        # ------------------------------------------
        # Class probabilities
        # ------------------------------------------

        pneumonia_probability = prediction
        normal_probability = 1 - prediction

        if prediction >= 0.5:

            predicted_class = "PNEUMONIA"
            confidence = pneumonia_probability

        else:

            predicted_class = "NORMAL"
            confidence = normal_probability

        # ------------------------------------------
        # Result
        # ------------------------------------------

        st.subheader("Prediction Result")

        if predicted_class == "PNEUMONIA":

            st.error(
                f"### Prediction: {predicted_class}"
            )

        else:

            st.success(
                f"### Prediction: {predicted_class}"
            )

        st.write(
            f"**Confidence: {confidence * 100:.2f}%**"
        )

        # ------------------------------------------
        # Probability
        # ------------------------------------------

        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "NORMAL",
                f"{normal_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "PNEUMONIA",
                f"{pneumonia_probability * 100:.2f}%"
            )

        st.progress(
            float(pneumonia_probability)
        )

    except Exception:

        st.error(
            "❌ Invalid image. Please upload a valid JPG, JPEG, or PNG chest X-ray."
        )
