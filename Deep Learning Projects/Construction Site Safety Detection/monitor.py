import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import time
import cv2
import tracking
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import Counter

# ------------------------------------------------------------
# MLflow Configuration
# ------------------------------------------------------------

tracking.set_tracking_uri("file:./mlruns")
tracking.set_experiment("Construction Site Safety Monitoring")

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = "Construction Site Safety/YOLO11/weights/best.pt"

IMAGE_FOLDER = "archive/valid/images"

CONF_THRESHOLD = 0.25

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "Machinery",
    "Vehicle"
]

# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

model = YOLO(MODEL_PATH)

# ------------------------------------------------------------
# Image List
# ------------------------------------------------------------

image_files = []

for file in os.listdir(IMAGE_FOLDER):

    if file.endswith((".jpg", ".jpeg", ".png")):

        image_files.append(os.path.join(IMAGE_FOLDER, file))

# ------------------------------------------------------------
# Monitoring Run
# ------------------------------------------------------------

with tracking.start_run(run_name="YOLO11_Model_Monitoring"):

    latency_list = []

    confidence_list = []

    class_counter = Counter()

    total_objects = 0

    for image_path in image_files:

        image = cv2.imread(image_path)

        height, width = image.shape[:2]

        start = time.time()

        results = model.predict(

            source=image,

            conf=CONF_THRESHOLD,

            verbose=False

        )

        end = time.time()

        latency = end - start

        latency_list.append(latency)

        boxes = results[0].boxes

        total_objects += len(boxes)

        for box in boxes:

            cls = int(box.cls.item())

            conf = float(box.conf.item())

            confidence_list.append(conf)

            class_counter[CLASS_NAMES[cls]] += 1

        # Log image information

        tracking.log_metric("Image Width", width)

        tracking.log_metric("Image Height", height)

    # --------------------------------------------------------
    # Aggregate Metrics
    # --------------------------------------------------------

    average_latency = np.mean(latency_list)

    max_latency = np.max(latency_list)

    min_latency = np.min(latency_list)

    average_confidence = np.mean(confidence_list) if confidence_list else 0

    confidence_std = np.std(confidence_list) if confidence_list else 0

    # --------------------------------------------------------
    # Log Metrics
    # --------------------------------------------------------

    tracking.log_metric("Images Processed", len(image_files))

    tracking.log_metric("Detected Objects", total_objects)

    tracking.log_metric("Average Latency", average_latency)

    tracking.log_metric("Maximum Latency", max_latency)

    tracking.log_metric("Minimum Latency", min_latency)

    tracking.log_metric("Average Confidence", average_confidence)

    tracking.log_metric("Confidence Std", confidence_std)

    # --------------------------------------------------------
    # Log Class Counts
    # --------------------------------------------------------

    for cls, count in class_counter.items():

        tracking.log_metric(f"{cls}_Count", count)

    # --------------------------------------------------------
    # Save Monitoring Report
    # --------------------------------------------------------

    report = pd.DataFrame({

        "Class": list(class_counter.keys()),

        "Detections": list(class_counter.values())

    })

    report.to_csv("monitoring_report.csv", index=False)

    tracking.log_artifact("monitoring_report.csv")

    # --------------------------------------------------------
    # Save Confidence Distribution
    # --------------------------------------------------------

    confidence_df = pd.DataFrame({

        "Confidence": confidence_list

    })

    confidence_df.to_csv(

        "confidence_distribution.csv",

        index=False

    )

    tracking.log_artifact(

        "confidence_distribution.csv"

    )

    # --------------------------------------------------------
    # Log Model Information
    # --------------------------------------------------------

    tracking.set_tags({

        "Project": "Construction Site Safety",

        "Framework": "YOLO11",

        "Author": "Amna Ameer",

        "Monitoring": "Inference",

        "Version": "1.0"

    })

    print("=" * 60)

    print("MODEL MONITORING REPORT")

    print("=" * 60)

    print(f"Images Processed : {len(image_files)}")

    print(f"Objects Detected : {total_objects}")

    print(f"Average Latency  : {average_latency:.4f} sec")

    print(f"Average Confidence : {average_confidence:.4f}")

    print("\nClass Distribution")

    for cls, count in class_counter.items():

        print(f"{cls:<20} {count}")

    print("=" * 60)