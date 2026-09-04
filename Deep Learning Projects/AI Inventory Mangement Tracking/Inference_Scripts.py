from predict import Predictor

predictor = Predictor("weights/best.pt")

predictor.predict(
    source="test.jpg",
    confidence=0.4
)