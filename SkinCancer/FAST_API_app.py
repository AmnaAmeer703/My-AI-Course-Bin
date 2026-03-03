# FastAPI + ResNet50 Model

# I used cat_img.jpg and another examples for testing (attached)
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import keras
from keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model("model.h5")

@app.post("/classify")
async def classify_image(image_file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image_file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")

        # Resize the image to 224x244, because I had a problem with the image size... so I tested (cat_img.jp)
        pil_image = pil_image.resize((224, 224))

        # Convert to array
        img_array = image.img_to_array(pil_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict the class of the image
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Format the predictions
        formatted_predictions = [{"label": str(pred[1]), "probability": float(pred[2])} for pred in decoded_predictions]

        return {"predictions": formatted_predictions}

    except Exception as e:
        print(f"Error during image processing and prediction, tray again please!: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5500)

### For testing API
# uvicorn app:app --reload
# http://localhost:8000/docs
