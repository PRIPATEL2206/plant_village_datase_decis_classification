'''
this is file that is made for farmerns for classifining decis in plant
'''

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
from utils.middeleWare import origins
from utils.response_utils import file_to_np_array, response_from_prediction

app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#models
potatoDecisClasificationModel =tf.keras.models.load_model("models/potatoes.h5")
potato_classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

pepperBellDecisClasificationModel =tf.keras.models.load_model("models/pepperBell.h5")
pepper_bell_classes=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']




# routes
@app.get("/")
def hello():
    return "hello"

@app.post("/api/potatoDecisClassifiction")
async def potatoDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await file_to_np_array(files)
        pred_arr=potatoDecisClasificationModel.predict(df)

        return response_from_prediction(pred_arr,potato_classes)
    
    except Exception as e:
        return {
            "error":str(e)
        }


@app.post("/api/pepperBellDecisClassifiction")
async def pepperBellDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await file_to_np_array(files)
        pred_arr=pepperBellDecisClasificationModel.predict(df)

        return response_from_prediction(pred_arr,pepper_bell_classes)
    
    except Exception as e:
        return {
            "error":str(e)
        }



if __name__=='__main__':
    uvicorn.run(app)