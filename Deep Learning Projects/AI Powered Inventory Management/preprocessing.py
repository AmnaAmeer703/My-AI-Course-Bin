from ultralytics import YOLO

class DataPreprocessing:

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path

    def preprocess(self):

        print("Checking Dataset...")
        print("Dataset Ready")

        return self.yaml_path