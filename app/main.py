from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pydantic import BaseModel
import numpy as np
import cv2

import asyncio
import base64

from .predict import predict_img

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def hello():
    await asyncio.sleep(1)  # Sleep for 5 seconds
    return JSONResponse(content={"content": "hello"})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        gesture_labels = {-1: "Too many hands", 0: 'A', 1: 'B', 2: 'L'}
    
        # Read image file, wait for read (do not cont with predict coroutine now)
        img_data = await file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #make it async
        prediction_result, prob = predict_img(img_rgb)

        if prob >= 0.8:
            predicted_label = gesture_labels[int(prediction_result)]
        else:
            predicted_label = "Not confident enough..."
        return {
            'prediction': predicted_label,
            'probability': prob
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class ImageData(BaseModel):
    imageSrc: str  # The base64-encoded image data
    letter:str

@app.post("/predict2")
def predict(image_data: ImageData):
    try:
        letter = image_data.letter
        gesture_labels = {-1: "Too many hands", 0: 'a', 1: 'b', 2: 'l'}
        
        # Remove the "data:image/jpeg;base64," prefix if it exists
        base64_data = image_data.imageSrc.replace("data:image/jpeg;base64,", "")

        # Decode the base64-encoded image data
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #make it async
        prediction_result, prob = predict_img(img_rgb)

        if prob >= 0.8:
            predicted_label = gesture_labels[int(prediction_result)]
        else:
            predicted_label = None

        if predicted_label == letter:
            message = f"Nice job, now it looks like you dominate letter {letter.upper()}, to return to chat click X on the right upper corner."
            success = True
        else:
            message = f"Mmmm this does not completely look like letter {letter.upper()}, but don't get upset you can try again by clicking retry button."
            success = False
        return {"content":message, "success":success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
