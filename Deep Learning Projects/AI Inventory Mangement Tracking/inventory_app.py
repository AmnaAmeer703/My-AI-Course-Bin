import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
from chatbot import ask_inventory_bot

st.set_page_config(
    page_title="Warehouse Inventory Management",
    page_icon="🤖",
    layout="wide"
)

st.write("Application Started")

st.title("Warehouse Inventory Management Monitoring")

# -------------------------------
# PATHS
# -------------------------------
ROOT = Path(__file__).resolve().parent

WEIGHTS_DIR = ROOT / "weights"

DETECTION_MODEL = WEIGHTS_DIR / "best.pt"
SEGMENTATION_MODEL = WEIGHTS_DIR / "yolo26n-seg.pt"
POSE_MODEL = WEIGHTS_DIR / "yolo26n-pose.pt"

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


from collections import defaultdict
# Global variables
class_counts = defaultdict(set)
inventory = {}
total = 0
# ==================================================
# IMAGE DETECTION
# ==================================================
if source == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert('RGB')

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
                    class_counts.clear()
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        class_counts[cls_id].add(len(class_counts[cls_id]))
                        st.write(f"Class: {model.names[cls_id]} | Confidence: {conf_score:.2f}"
                                 )
                        total = sum(len(v) for v in class_counts.values())
                    
# Video Tracking Through ByteTrack
# from collections import defaultdict

elif source == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Tracking"):

            temp_video = "temp_video.mp4"

            with open(temp_video, "wb") as f:
                f.write(uploaded_video.read())

            cap = cv2.VideoCapture(temp_video)

            frame_placeholder = st.empty()

            class_counts = defaultdict(set)

            while cap.isOpened():

                success, frame = cap.read()

                if not success:
                    break

                results = model.track(
                    frame,
                    conf=confidence,
                    persist=True,
                    tracker="bytetrack.yaml"
                )

                result = results[0]

                annotated = result.plot()

                if result.boxes.id is not None:

                    ids = result.boxes.id.cpu().numpy().astype(int)
                    classes = result.boxes.cls.cpu().numpy().astype(int)

                    for obj_id, cls in zip(ids, classes):
                        class_counts[cls].add(obj_id)

                y = 30

                total = 0

                for cls in sorted(class_counts.keys()):

                    count = len(class_counts[cls])
                    total += count

                    cv2.putText(
                        annotated,
                        f"{model.names[cls]} : {count}",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2
                    )

                    y += 30

                cv2.putText(
                    annotated,
                    f"Total Objects : {total}",
                    (20, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )

                frame_placeholder.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )

            cap.release()

elif source == "Webcam":

    run = st.checkbox("Start Webcam")

    frame_placeholder = st.empty()

    class_counts = defaultdict(set)

    if run:

        cap = cv2.VideoCapture(0)

        while run:

            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                frame,
                conf=confidence,
                persist=True,
                tracker="bytetrack.yaml"
            )

            result = results[0]

            annotated = result.plot()

            if result.boxes.id is not None:

                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)

                for obj_id, cls in zip(ids, classes):
                    class_counts[cls].add(obj_id)

            y = 30
            total = 0

            for cls in sorted(class_counts):

                count = len(class_counts[cls])
                total += count

                cv2.putText(
                    annotated,
                    f"{model.names[cls]} : {count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

                y += 30

            cv2.putText(
                annotated,
                f"Total : {total}",
                (20, y+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )

            frame_placeholder.image(
                annotated,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

# Inventory Table

st.sidebar.subheader("Live Inventory")

if class_counts:
    inventory = {
        model.names[cls]: len(obj_ids)
        for cls, obj_ids in class_counts.items()
    }
    st.sidebar.subheader("📦 Live Inventory")
if inventory:
    st.sidebar.dataframe(inventory)
else:
    st.sidebar.info("No inventory detected yet.")

# KPI Cards

total = sum(len(v) for v in class_counts.values())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Inventory", total)
col2.metric("Unique Classes", len(class_counts))

if source == "Image":
    col3.metric("Mode", "Detection")
else:
    col3.metric("Tracking", "ByteTrack")

col4.metric("Confidence", f"{confidence:.2f}")


# ===================================

# ===================================
# AI INVENTORY ASSISTANT
# ===================================

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Inventory Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.sidebar.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.sidebar.text_input("Ask about inventory")

if st.sidebar.button("Send", use_container_width=True):

    if question.strip():

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.spinner("Thinking..."):
            answer = ask_inventory_bot(
                question,
                inventory,
                source
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        st.rerun()

st.sidebar.success("✅ Sidebar reached")