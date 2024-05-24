'''
this is file that is made for farmerns for classifining decis in plant
'''

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
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

########################################## constants  #######################################################################
plants=["pepperBell","potato","tomato"]


########################################## Models ###########################################################################

models={
        "plantClasificationModel" :tf.keras.models.load_model("models/plantClasificationModelWithImbalance_accurecy_0.9957_epohs_20_.h5"),
        plants[0] :tf.keras.models.load_model("models/pepperBell.h5"),
        plants[1] :tf.keras.models.load_model("models/potatoes.h5"),
        plants[2] :tf.keras.models.load_model("models/tomatoPlant.h5")
}

######################################### desisclasies ####################################################################

desis={
    plants[0]:['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy'],
    plants[1]:['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    plants[2]:['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot','Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
}


############################################ Solutions ##################################################################
solutions={
    plants[0]:[
        "Remove affected parts of plant and take them far away from the garden",
        "No Need To Do any thing"
        ],
    plants[1]:[
        "Rise Fungicide/Nematicide have been shown to protect potatoes from early blight and other foliar fungal disease",
        "Your best bet it to remove and destroy infected plants, along with any tubers or plant material in the soil that may be infected.",
        "No Need To Do any thing"
        ],
    plants[2]:[
        "Copper fungicides are the most commonly recommended treatment for bacterial leaf spot.",
        "Construct narrow Trenches alongside your tomato plant to ensure that water enters the plant’s root.",
        "Strategies for managing late blight in tomato include planting resistant cultivars, eliminating volunteers spacing plants to increase airflow and reduce humidity, and applying preventive and effective fungicides to avoid infection.",
        "Remove and destroy all crop debris post-harvest. Sanitize the greenhouse between crop seasons. Use fans and avoid overhead watering to minimize leaf wetness.",
        "Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease.",
        "In the intricate symphony of gardening, where tomato plants sway to the rhythms of the seasons, the presence of red spider mites can strike a discordant note. ",
        "The products to use are chlorothalonil, copper oxychloride or mancozeb. Treatment should start when the first spots are seen and continue at 10-14-day intervals until 3-4 weeks before last harvest",
        "Currently, the most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties",
        "You may practice good sanitation practices to prevent the spread of infection to the rest of the garden",
        "No Need To Do any thing",
    ]
}


################################################## Functions for diffrent plants #####################################################

def classfyDesisUsingModel(df,model,classes,solutions,files,return_response=True): 
    pred_arr=model.predict(df)
    if not return_response:
        return pred_arr
    return response_from_prediction(pred_arr,classes,files,solutions=solutions)
    
    


############################  Routes  ##############################################
@app.get("/")
def hello():
    return "hello"



@app.post("/api/plantDecisClasification")
async def plantDesisClassification(files:list[UploadFile] = File(...)):
    try:
        df= await files_to_np_array(files)
        preds= classfyDesisUsingModel(
            df=df,
            model=models["plantClasificationModel"],
            classes=plants,
            files=files,
            solutions=plants,
            return_response=False
            )
        print(np.array([df[0]]).shape)
        pred_plants=[plants[ np.argmax(pred)] for pred in preds]

        response=[]
        for i, pred_plant in enumerate(pred_plants):
            prediction_desis = classfyDesisUsingModel(
                df=np.array([df[i]]),
                model=models[pred_plant],
                classes=desis[pred_plant],
                files=[files[i]],
                solutions=solutions[pred_plant],
            )
            prediction_desis[0]["plant_confidence"]={i:round(float(j) ,4) for i,j in sorted(zip(plants,preds[i]), key= lambda x: x[1],reverse=True)}
            response.append(prediction_desis[0])
        return response
    except Exception as e:
        print(e)
        return {
            "error":str(e)
        }



@app.post("/api/potatoDecisClassifiction")
async def potatoDesisClasification(files:list[UploadFile] = File()):
    try:
        df= await files_to_np_array(files)
        return classfyDesisUsingModel(
            df=df,
            model=models["potato"],
            classes=desis["potato"],
            solutions=solutions["potato"],
            files=files
            )
    except Exception as e:
        return {
            "error":str(e)
        }
    


@app.post("/api/pepperBellDecisClassifiction")
async def pepperBellDesisClasification(files:list[UploadFile] = File(...)):
    try:
        df= await files_to_np_array(files)
        return classfyDesisUsingModel(
            df=df,
            model=models["pepperBell"],
            classes=desis["pepperBell"],
            solutions=solutions["pepperBell"],
            files=files
            )
    
    except Exception as e:
        return {
            "error":str(e)
        }



@app.post("/api/tomatoDecisClassifiction")
async def tomatoDesisClasification(files:list[UploadFile] = File(...)):
    try:

        df= await files_to_np_array(files)
        return classfyDesisUsingModel(
            df=df,
            model=models["tomato"],
            classes=desis["tomato"],
            solutions=solutions["tomato"],
            files=files
            )
    
    except Exception as e:
        return {
            "error":str(e)
        }



if __name__=='__main__':
    uvicorn.run(app)