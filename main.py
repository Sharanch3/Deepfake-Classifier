import os
import uuid
import streamlit as st
from utils import predict

st.set_page_config(
    page_title="DeepFake | Image Classifier",
    page_icon="🎭",
    layout="centered"
)

# Header
st.title("🎭 DeepFake Image Classifier")
st.caption("Upload an image to detect whether it's real or AI-generated.")
st.divider()

uploaded_file = st.file_uploader(
    label="Choose an image to analyze",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    # Display image
    st.image(uploaded_file, caption=f"📁 {uploaded_file.name}", use_container_width=True)
    st.divider()

    # Use a unique filename to avoid collisions
    ext = os.path.splitext(uploaded_file.name)[-1]
    temp_path = f"temp_{uuid.uuid4().hex}{ext}"

    try:
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("🔍 Analyzing image..."):
            prediction = predict(temp_path)

        # Result
        st.subheader("Analysis Result")
        if prediction == "REAL":
            st.success("✅ **Real Image** — This image appears to be authentic.")
        else:
            st.warning("🎭 **AI Generated** — This image appears to be synthetically generated.")

        # Confidence hint (if predict returns a label+score dict, expand here)
        with st.expander("ℹ️ About this result"):
            st.write(
                "This classifier uses a deep learning model to distinguish between "
                "real photographs and AI-generated images. Results may not be 100% accurate."
            )

    except Exception as e:
        st.error(f"❌ Something went wrong during analysis: {e}")

    finally:
        # Always clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

else:
    st.info("🖼️ Upload a JPG or PNG image to get started.")

