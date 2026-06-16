import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tempfile
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

st.title("Traffic Violation YOLO Detection")

# =========================
# IMAGE DETECTION
# =========================

st.header("Traffic Violation Detection")

image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image"
)

if image_file is not None:
    image = np.array(Image.open(image_file))

    results = model(image)
    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        channels="BGR",
        caption="Detected Objects"
    )

# =========================
# VIDEO DETECTION
# =========================

st.header("Video Detection")

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"],
    key="video"
)

if video_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)

    frame_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        frame_placeholder.image(
            annotated_frame,
            channels="BGR"
        )

    cap.release()

# =========================
# WEBCAM DETECTION
# =========================

st.header("Webcam Detection")

camera_image = st.camera_input("Take a photo")

if camera_image is not None:

    image = np.array(Image.open(camera_image))

    results = model(image)
    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        channels="BGR",
        caption="Webcam Detection Result"
    )




