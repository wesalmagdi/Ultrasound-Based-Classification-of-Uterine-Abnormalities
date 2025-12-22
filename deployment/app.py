import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Uterine AI Diagnostic Tool",
    page_icon="🩺",
    layout="centered"
)

# --- 1. Load the Model ---
@st.cache_resource
def load_hybrid_model():
    # Loading the model you saved (which expects image + tabular inputs)
    # We use compile=False to avoid needing custom loss function definitions
    return tf.keras.models.load_model('vgg16_final.h5', compile=False)

model = load_hybrid_model()

# --- 2. UI Header ---
st.title("🩺 Uterine Abnormality Predictor")
st.write("Upload an ultrasound scan for an automated AI diagnosis.")
st.info("System: Image-Only Input Mode (Hybrid Architecture)")

# --- 3. Image Upload ---
uploaded_file = st.file_uploader("Choose an ultrasound image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Ultrasound Scan', use_column_width=True)
    
    # --- 4. Prediction Button ---
    if st.button("Generate Diagnostic Result"):
        with st.spinner('Analyzing medical imaging data...'):
            try:
                # A. Image Preprocessing
                # 1. Convert to RGB (VGG16 standard)
                img = image.convert('RGB')
                # 2. Resize to 224x224 (The size your error message indicated)
                img = img.resize((224, 224))
                # 3. Normalize pixels to [0, 1]
                img_array = np.array(img) / 255.0
                # 4. Expand dims to create batch size: (1, 224, 224, 3)
                img_input = np.expand_dims(img_array, axis=0)

                # B. Tabular "Dummy" Data
                # Your model 'hybrid_vgg16_tabular' expects 2 inputs.
                # We provide 400 zeros to satisfy the second input layer.
                tabular_input = np.zeros((1, 40))

                # C. Run Prediction
                # We pass the list of two inputs as required by the Layer specs
                prediction_raw = model.predict([img_input, tabular_input])
                prediction = prediction_raw[0][0]

                # --- 5. Display Results ---
                st.divider()
                
                if prediction > 0.5:
                    st.error("### RESULT: ABNORMALITY DETECTED")
                    st.write(f"**Confidence Level:** {prediction:.2%}")
                    st.warning("Recommendation: This result suggests an abnormality. Please refer to clinical findings and radiologist review.")
                else:
                    st.success("### RESULT: NORMAL / NO ABNORMALITY")
                    st.write(f"**Confidence Level:** {(1 - prediction):.2%}")
                    st.info("Recommendation: No immediate abnormality detected by the AI. Periodic follow-up is advised.")

            except Exception as e:
                st.error("An error occurred during processing.")
                st.info(f"Technical details: {e}")

# --- 6. Footer ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational/research purposes only and should not replace professional medical advice.")