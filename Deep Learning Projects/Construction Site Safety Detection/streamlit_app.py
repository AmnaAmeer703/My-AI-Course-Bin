import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
st.write("Application Started")
# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Construction Site Safety Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("Construction Site Safety")

# -------------------------------
# PATHS
# -------------------------------
ROOT = Path(__file__).parent

WEIGHTS_DIR = ROOT / "weights"

DETECTION_MODEL = WEIGHTS_DIR / "best.pt"
SEGMENTATION_MODEL = WEIGHTS_DIR / "yolo11s-seg.pt"
POSE_MODEL = WEIGHTS_DIR / "yolo11s-pose.pt"

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection", "Segmentation", "Pose Estimation"]
)

confidence = st.sidebar.slider(
    "Confidence",
    min_value=0.25,
    max_value=1.0,
    value=0.40
)

# -------------------------------
# LOAD MODEL
# -------------------------------
try:
    if model_type == "Detection":
        model = YOLO(str(DETECTION_MODEL))

    elif model_type == "Segmentation":
        model = YOLO(str(SEGMENTATION_MODEL))

    else:
        model = YOLO(str(POSE_MODEL))

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# -------------------------------
# SOURCE SELECTION
# -------------------------------
source = st.sidebar.radio(
    "Select Source",
    ["Image", "Video", "Webcam"]
)

# ==================================================
# IMAGE DETECTION
# ==================================================
if source == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        if st.button("Detect Objects"):

            image_np = np.array(image)

            results = model.predict(
                image_np,
                conf=confidence
            )

            annotated = results[0].plot()

            with col2:
                st.image(
                    annotated,
                    caption="Detection Result",
                    channels="BGR",
                    use_container_width=True
                )

            with st.expander("Detection Results"):
                boxes = results[0].boxes

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf_score = float(box.conf[0])

                    st.write(
                        f"Class: {model.names[cls_id]} | Confidence: {conf_score:.2f}"
                    )

# ==================================================
# VIDEO DETECTION
# ==================================================
elif source == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Detection"):

            temp_video = "temp_video.mp4"

            with open(temp_video, "wb") as f:
                f.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_video)

            frame_placeholder = st.empty()

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                results = model.predict(
                    frame,
                    conf=confidence
                )

                annotated_frame = results[0].plot()

                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True
                )

            cap.release()

# ==================================================
# WEBCAM DETECTION
# ==================================================
elif source == "Webcam":

    run = st.checkbox("Start Webcam")

    frame_placeholder = st.empty()

    if run:

        cap = cv2.VideoCapture(0)

        while run:

            success, frame = cap.read()

            if not success:
                st.error("Cannot access webcam.")
                break

            results = model.predict(
                frame,
                conf=confidence
            )

            annotated_frame = results[0].plot()

            frame_placeholder.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()
