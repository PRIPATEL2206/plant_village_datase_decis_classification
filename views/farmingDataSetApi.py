
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
model =tf.keras.models.load_model("models/potatoes.h5")
potato_classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


def image_to_nparry(img):
    img=Image.open(BytesIO(img)).convert("RGB").resize((256,256))
    return np.array(img)


@app.get("/")
def hello():
    return "hello"

@app.post("/api/potatoDecisClassifiction")
async def potatoDesisClasification(files:list[UploadFile] = File(...)):
    df= np.array([image_to_nparry(await file.read()) for file in files ])
    pred_arr=model.predict(df)

    return [ {
        "class":potato_classes[np.argmax(pred)],
        "confidence":float(pred.max())
        } for pred in pred_arr]


if __name__=='__main__':
    uvicorn.run(app)