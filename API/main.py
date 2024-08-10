from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf 
import os

app = FastAPI()

model_path = os.path.abspath("../Models/2")

MODEL = tf.keras.layers.TFSMLayer(model_path)

# MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ["Potato_Early_Blight","Potato_Late_Blight","Potato_Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am Alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    image_batch = np.expand_dims(image,axis=0)
    image_batch = tf.cast(image_batch,tf.float32)

    prediction = MODEL(image_batch)  

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {
        'Predicted Class':predicted_class,
        'Confidence':float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)