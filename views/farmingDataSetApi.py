
from io import BytesIO
from  PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import uvicorn

app=FastAPI()

origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# (32, 923, 4) gain
# (32, 256, 256, 3) expected
# (634, 923, 4)
model =tf.keras.models.load_model("potatoes.h5")
potato_classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def image_to_nparry(img):
    img=Image.open(BytesIO(img)).convert("RGB").resize((256,256))
    return np.array(img)


@app.get("/")
def hello():
    return "hello"

@app.post("/api/potatoDecisClassifiction")
async def potatoDesisClasification(file:UploadFile = File(...)):
    df= np.expand_dims(image_to_nparry(await file.read()),0)
    predArr=model.predict(df)
    pred_class=potato_classes[np.argmax(predArr)]
    return {
        "class":pred_class,
        "confidence":float(predArr.max())
        }


# if __name__=='__main__':
#     uvicorn.run(app)