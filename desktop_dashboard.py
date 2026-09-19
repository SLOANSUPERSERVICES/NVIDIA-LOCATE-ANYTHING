import streamlit as st
import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont

# Ensure the Embodied directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "Embodied"))

try:
    from locateanything_worker import LocateAnythingWorker
except ImportError as e:
    st.error(f"Failed to import LocateAnythingWorker. Error: {e}")
    st.stop()

st.set_page_config(page_title="Mesh Claw Assistant 🦀", layout="wide")

@st.cache_resource
def load_model():
    # Only initializing on CPU here for general compatibility, but users can switch to CUDA
    return LocateAnythingWorker("nvidia/LocateAnything-3B", device="cpu", dtype=torch.float32)

st.title("🦀 Mesh Claw Assistant")
st.subheader("Your visual workstation guide with superior vision capabilities.")
st.markdown("Upload a screenshot or use a connected camera, and ask the Claw to find UI components, text, or objects!")

# Load model
with st.spinner("Waking up the Claw (Loading Model)..."):
    try:
        if os.environ.get("TEST_LOAD", "0") == "1":
            worker = None
        else:
            worker = load_model()
    except Exception as e:
        st.error(f"Model load error: {e}")
        worker = None

# Input selection
input_method = st.radio("Choose Input Method", ["Upload Screenshot", "Use Camera"])

uploaded_file = None
if input_method == "Upload Screenshot":
    uploaded_file = st.file_uploader("Upload Desktop Screenshot...", type=["jpg", "jpeg", "png"])
else:
    # This will hook into whatever default webcam the browser has access to
    # If the Arduino camera exposes itself as a standard UVC webcam to the OS, it can work here.
    uploaded_file = st.camera_input("Take a picture with your connected camera")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(image, caption="Current View", use_column_width=True)

    with col2:
        st.markdown("### Command the Claw")
        query = st.text_input("What should I locate?", placeholder="e.g., 'the submit button', 'red apple'")
        action_type = st.radio("Action Type", ["GUI Element (Box)", "GUI Element (Point)", "Detect Text", "Object Detection"])

        if st.button("Locate!"):
            if not worker:
                st.warning("Model is running in dummy mode (TEST_LOAD=1).")
                st.stop()

            if not query and action_type != "Detect Text":
                st.warning("Please enter what you want to locate.")
            else:
                with st.spinner("The Claw is scanning..."):
                    w, h = image.size
                    draw = ImageDraw.Draw(image)

                    if action_type == "GUI Element (Box)":
                        result = worker.ground_gui(image, query, output_type="box")
                        boxes = worker.parse_boxes(result["answer"], w, h)
                        for box in boxes:
                            draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']], outline="red", width=3)

                    elif action_type == "GUI Element (Point)":
                        result = worker.ground_gui(image, query, output_type="point")
                        points = worker.parse_points(result["answer"], w, h)
                        for pt in points:
                            r = 5
                            draw.ellipse([pt['x']-r, pt['y']-r, pt['x']+r, pt['y']+r], fill="red", outline="white", width=2)

                    elif action_type == "Object Detection":
                        result = worker.detect(image, [query])
                        boxes = worker.parse_boxes(result["answer"], w, h)
                        for box in boxes:
                            draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']], outline="green", width=3)

                    elif action_type == "Detect Text":
                        result = worker.detect_text(image)
                        boxes = worker.parse_boxes(result["answer"], w, h)
                        for box in boxes:
                            draw.rectangle([box['x1'], box['y1'], box['x2'], box['y2']], outline="blue", width=2)

                    st.image(image, caption="Claw Vision Active", use_column_width=True)
                    st.success("Analysis Complete!")
