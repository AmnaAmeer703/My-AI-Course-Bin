import tracking
from mlflow.tracking import MlflowClient

# ==========================================================
# MLflow Configuration
# ==========================================================

tracking.set_tracking_uri("file:./mlruns")

MODEL_NAME = "Warehouse Inventory Management"

client = MlflowClient()

# ==========================================================
# Get Latest Run
# ==========================================================

experiment = tracking.get_experiment_by_name(
    "Warehouse Inventory Management"
)

if experiment is None:
    raise Exception("Experiment not found.")

experiment_id = experiment.experiment_id

runs = tracking.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if runs.empty:
    raise Exception("No training runs found.")

run_id = runs.iloc[0]["run_id"]

print(f"Latest Run ID : {run_id}")

# ==========================================================
# Model URI
# ==========================================================

model_uri = f"runs:/{run_id}/model"

print(f"Model URI : {model_uri}")

# ==========================================================
# Register Model
# ==========================================================

registered_model = tracking.register_model(

    model_uri=model_uri,

    name=MODEL_NAME

)

print(f"\nModel Registered Successfully!")

print(f"Version : {registered_model.version}")

# ==========================================================
# Add Version Description
# ==========================================================

client.update_model_version(

    name=MODEL_NAME,

    version=registered_model.version,

    description="""
YOLO11 Construction Site Safety Detection Model

Classes:
- Hardhat
- Mask
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest
- Machinery
- Vehicle

Author:
Amna Ameer
"""
)

# ==========================================================
# Add Model Description
# ==========================================================

client.update_registered_model(

    name=MODEL_NAME,

    description="""
Construction Site Safety Detection

This model detects:

• PPE Compliance
• Workers
• Machinery
• Vehicles
• Safety Cones

Framework:
Ultralytics YOLO11

Author:
Amna Ameer
"""
)

# ==========================================================
# Assign Alias
# ==========================================================

client.set_registered_model_alias(

    MODEL_NAME,

    "champion",

    registered_model.version

)

print("Alias 'champion' assigned.")

# ==========================================================
# Add Tags
# ==========================================================

client.set_model_version_tag(

    MODEL_NAME,

    registered_model.version,

    "Framework",

    "YOLO11"

)

client.set_model_version_tag(

    MODEL_NAME,

    registered_model.version,

    "Dataset",

    "Construction Site Safety"

)

client.set_model_version_tag(

    MODEL_NAME,

    registered_model.version,

    "Author",

    "Amna Ameer"

)

client.set_model_version_tag(

    MODEL_NAME,

    registered_model.version,

    "Task",

    "Object Detection"

)

# ==========================================================
# Display All Versions
# ==========================================================

print("\nRegistered Versions")

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

for version in versions:

    print("-" * 40)

    print("Version :", version.version)

    print("Current Stage :", version.current_stage)

    print("Status :", version.status)

    print("Run ID :", version.run_id)

print("-" * 40)

print("\nRegistry Completed Successfully.")