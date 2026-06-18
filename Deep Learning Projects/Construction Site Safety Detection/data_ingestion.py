import yaml
from pathlib import Path

class DataIngestion:

    def __init__(self, yaml_path):
        self.yaml_path = yaml_path

    def load_yaml(self):

        with open(self.yaml_path) as f:
            data = yaml.safe_load(f)

        return data

    def show_dataset(self):

        data = self.load_yaml()

        print("Training :", data["train"])
        print("Validation :", data["val"])
        print("Testing :", data["test"])
        print("Classes :", data["names"])