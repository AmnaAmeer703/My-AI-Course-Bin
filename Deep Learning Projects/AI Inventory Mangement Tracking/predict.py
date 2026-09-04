from ultralytics import YOLO

class Predictor:

    def __init__(self, weight_path):

        self.model = YOLO(weight_path)

    def predict(self,
                source,
                confidence):

        results = self.model.predict(
            source=source,
            conf=confidence,
            save=True
        )

        return results