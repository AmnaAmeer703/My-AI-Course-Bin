from config import *

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessing
from train import ModelTrainer
from evaluate import Evaluator

def run_pipeline():

    print("="*50)
    print("Warehouse Inventory Management")
    print("="*50)

    ingestion = DataIngestion(YAML_FILE)

    ingestion.show_dataset()

    preprocess = DataPreprocessing(YAML_FILE)

    dataset = preprocess.preprocess()

    trainer = ModelTrainer(MODEL_NAME)

    trainer.train(
        yaml_file=dataset,
        epochs=TRAIN_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE
    )

    evaluator = Evaluator("runs/detect/train/weights/best.pt")

    evaluator.evaluate()

    print("Pipeline Completed Successfully")


if __name__ == "__main__":

    run_pipeline()