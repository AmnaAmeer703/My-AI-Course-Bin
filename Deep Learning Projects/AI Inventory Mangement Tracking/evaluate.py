from ultralytics import YOLO

class Evaluator:

    def __init__(self, weight_path):

        self.model = YOLO(weight_path)

    def evaluate(self):

        metrics = self.model.val()

        print(metrics)