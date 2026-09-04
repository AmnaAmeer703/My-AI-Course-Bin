import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import time
import shutil
import mlflow
from mlflow import pyfunc
from mlflow.models.signature import infer_signature
from ultralytics import YOLO
import pandas as pd
import numpy as np

# ==========================================================
# MLflow Configuration
# ==========================================================



mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Construction Site Safety Detection")

# ==========================================================
# Dataset
# ==========================================================
from pathlib import Path
ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "archive"/"css-data"
YAML_FILE = ROOT / "working"/"data.yaml"

WEIGHTS = ROOT / "weights"

MODEL_NAME = "yolo11s.pt"

PROJECT_NAME = "Construction Site Safety"

RUN_NAME = "YOLO11_Construction_Safety"

# ==========================================================
# Hyperparameters
# ==========================================================

params = {
    "epochs": 30,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 2,
    "optimizer": "auto",
    "lr0": 0.01,
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.1,
    "fliplr": 0.5,
    "scale": 0.5,
    "translate": 0.1,
    "degrees": 15,
    "shear": 10,
    "perspective": 0.0005
}
import torch

DEVICE = 0 if torch.cuda.is_available() else "cpu"

params["device"] = DEVICE

# ==========================================================
# Start MLflow
# ==========================================================

with mlflow.start_run(run_name=RUN_NAME):

    start = time.time()

    #########################################################
    # Log Parameters
    #########################################################

    mlflow.log_params(params)

    #########################################################
    # Load Model
    #########################################################

    model = YOLO(MODEL_NAME)

    #########################################################
    # Train
    #########################################################
    print("=" * 50)
    print("Current working directory:", os.getcwd())
    print("DATA_YAML:", YAML_FILE)
    with open(YAML_FILE, "r") as f:
        print(f.read())
        print("=" * 50)
    ########################################################


    model.train(

        data=YAML_FILE,

        epochs=params["epochs"],

        imgsz=params["imgsz"],

        batch=params["batch"],

        device=params["device"],

        workers=params["workers"],

        optimizer=params["optimizer"],

        lr0=params["lr0"],

        mosaic=params["mosaic"],

        mixup=params["mixup"],

        copy_paste=params["copy_paste"],

        fliplr=params["fliplr"],

        scale=params["scale"],

        translate=params["translate"],

        degrees=params["degrees"],

        shear=params["shear"],

        perspective=params["perspective"],

        project=PROJECT_NAME,

        name="YOLO11",

        exist_ok=True
    )

    #########################################################
    # Validation
    #########################################################

    metrics = model.val()

    #########################################################
    # Log Metrics
    #########################################################

    mlflow.log_metric("mAP50", metrics.box.map50)

    mlflow.log_metric("mAP50_95", metrics.box.map)

    mlflow.log_metric("Precision", metrics.box.mp)

    mlflow.log_metric("Recall", metrics.box.mr)

    #########################################################
    # Training Time
    #########################################################

    end = time.time()

    mlflow.log_metric("Training_Time(seconds)", end-start)

    #########################################################
    # Artifact Paths
    #########################################################

    run_folder = os.path.join(
        PROJECT_NAME,
        "YOLO11"
    )

    artifact_files = [

        "results.png",

        "confusion_matrix.png",

        "PR_curve.png",

        "P_curve.png",

        "R_curve.png",

        "F1_curve.png",

        "labels.jpg"

    ]

    #########################################################
    # Log Images
    #########################################################

    for file in artifact_files:

        path = os.path.join(run_folder, file)

        if os.path.exists(path):

            mlflow.log_artifact(path)

    #########################################################
    # Best Model
    #########################################################

    best_model = os.path.join(

        run_folder,

        "weights",

        "best.pt"

    )

    if os.path.exists(best_model):

        mlflow.log_artifact(best_model)

    #########################################################
    # ONNX Export
    #########################################################

    best = YOLO(best_model)

    best.export(format="onnx")

    onnx_model = os.path.join(

        run_folder,

        "weights",

        "best.onnx"

    )

    if os.path.exists(onnx_model):

        mlflow.log_artifact(onnx_model)

    #########################################################
    # Dummy Signature
    #########################################################

    input_example = pd.DataFrame(

        np.random.rand(1,640),

        columns=[f"pixel_{i}" for i in range(640)]

    )

    output_example = pd.DataFrame(

        {"Prediction":[0]}

    )

    signature = infer_signature(

        input_example,

        output_example

    )

    #########################################################
    # Log PyFunc Model
    #########################################################

    class YOLOWrapper(mlflow.pyfunc.PythonModel):

        def load_context(self, context):

            self.model = YOLO(context.artifacts["model"])

        def predict(self, context, model_input):

            return self.model(model_input)

    mlflow.pyfunc.log_model(

        artifact_path="model",

        python_model=YOLOWrapper(),

        artifacts={"model": best_model},

        signature=signature,

        input_example=input_example,

        registered_model_name="Construction_Site_Safety_YOLO11"

    )

    #########################################################
    # Tags
    #########################################################

    mlflow.set_tags({

        "Framework": "Ultralytics YOLO11",

        "Task": "Object Detection",

        "Dataset": "Construction Site Safety",

        "Author": "Amna Ameer"

    })

    print("="*60)

    print("Training Completed Successfully")

    print(f"mAP50      : {metrics.box.map50:.4f}")

    print(f"mAP50-95   : {metrics.box.map:.4f}")

    print(f"Precision  : {metrics.box.mp:.4f}")

    print(f"Recall     : {metrics.box.mr:.4f}")

    print("="*60)