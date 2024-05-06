'''
this is file that is made for farmerns for classifining decis in plant
'''

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
from utils.middeleWare import origins
from utils.response_utils import files_to_np_array, response_from_prediction

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

tomatoDecisClasificationModel =tf.keras.models.load_model("models/tomatoPlant.h5")
tomato_classes=['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']



# routes
@app.get("/")
def hello():
    return "hello"

@app.post("/api/potatoDecisClassifiction")
async def potatoDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await files_to_np_array(files)
        pred_arr=potatoDecisClasificationModel.predict(df)
        file_names=[i.filename for i in files]
        return response_from_prediction(pred_arr,potato_classes,file_names)
    
    except Exception as e:
        return {
            "error":str(e)
        }


@app.post("/api/pepperBellDecisClassifiction")
async def pepperBellDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await files_to_np_array(files)
        pred_arr=pepperBellDecisClasificationModel.predict(df)
        file_names=[i.filename for i in files]
        return response_from_prediction(pred_arr,pepper_bell_classes,file_names)
    
    except Exception as e:
        return {
            "error":str(e)
        }


@app.post("/api/tomatoDecisClassifiction")
async def tomatoDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await files_to_np_array(files)
        pred_arr=tomatoDecisClasificationModel.predict(df)
        file_names=[i.filename for i in files]
        return response_from_prediction(pred_arr,tomato_classes,file_names)
    
    except Exception as e:
        return {
            "error":str(e)
        }




if __name__=='__main__':
    uvicorn.run(app)