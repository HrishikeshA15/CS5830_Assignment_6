
#Importing Necessary Libraries
import os
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List
from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.models import Sequential

#step 1
app = FastAPI()

def load_model(path: str) -> Sequential:
    """
    Load the model from the supplied path on the disk and return the keras.src.engine.sequential.Sequential model.
    """
    if os.path.exists(path):
        from keras.models import load_model
        return load_model(path)
    else:
        return None
    
def predict_digit(model: Sequential, data_point: List[float]) -> str:
    """
    Take the image serialized as an array of 784 elements and returns the predicted digit as string.
    """
    data_point = np.array(data_point).reshape(1, 784)
    prediction = model.predict(data_point).argmax()
    return str(prediction)


    
@app.get('/')
def index():
    return {'message': "Welcome to the Digit Prediction API!"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI model loading")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()

    app.model_path = args.model_path
    app.model = load_model(app.model_path)
    
    if app.model is None:
        print(f"Error: Model file not found at {app.model_path}")
    else:
        print(f"Model loaded successfully from {app.model_path}")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Convert the image to grayscale and resize to 28x28
    img = Image.open(BytesIO(contents)).convert('L').resize((28, 28))
    # Convert image to array
    img_array = np.array(img).astype('float32') / 255.0
    # Flatten the array
    img_flat = img_array.flatten().tolist()
    # Load the model
    model = load_model("mnist_model.keras")
    # Make prediction
    prediction = predict_digit(model, img_flat)
    return {"digit": prediction}

#Task 2

def format_image(image):
    # Convert the image to grayscale and resize to 28x28
    image_grey = image.convert('L').resize((28, 28))
    # Serialize the image as an array of 784 elements
    serial_array = list(image_grey.getdata())
    return serial_array

@app.post("/predictany")
async def predict(file: UploadFile = File(...)):
    # Read the bytes from the uploaded image
    # await file.read()

    # Convert the bytes to a PIL Image
    image = Image.open(file.file)

    Serialized_image = format_image(image)
    
    # Load the model
    model = load_model("mnist_model.keras")
    
    # Get the predicted digit
    digit = predict_digit(model, Serialized_image)

    # Return the predicted digit
    return {"digit": digit}

# if __name__ == '__main__':
#     uvicorn.run(app, host="0.0.0.0", port=8000)