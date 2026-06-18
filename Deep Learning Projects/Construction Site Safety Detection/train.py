from ultralytics import YOLO

class ModelTrainer:

    def __init__(self, model_name):

        self.model = YOLO(model_name)

    def train(self,
              yaml_file,
              epochs,
              imgsz,
              batch):

        self.model.train(
            data=yaml_file,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch
        )

        print("Training Finished")