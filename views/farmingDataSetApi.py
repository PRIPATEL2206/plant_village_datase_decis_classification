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
potato_solutions=[
    "Rise Fungicide/Nematicide have been shown to protect potatoes from early blight and other foliar fungal disease",
     "Your best bet it to remove and destroy infected plants, along with any tubers or plant material in the soil that may be infected.",
     "No Need To Do any thing"
]

pepperBellDecisClasificationModel =tf.keras.models.load_model("models/pepperBell.h5")
pepper_bell_classes=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']
pepper_bell_solutions=[
    "Remove affected parts of plant and take them far away from the garden",
    "No Need To Do any thing"
    ]

tomatoDecisClasificationModel =tf.keras.models.load_model("models/tomatoPlant.h5")
tomato_classes=['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
tomato_solutions=[
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

# [
#     "The products to use are chlorothalonil, copper oxychloride or mancozeb. Treatment should start when the first spots are seen and continue at 10-14-day intervals until 3-4 weeks before last harvest",
#     "You may practice good sanitation practices to prevent the spread of infection to the rest of the garden",
#     "Currently, the most effective treatments used to control the spread of TYLCV are insecticides and resistant crop varieties",
#     "Copper fungicides are the most commonly recommended treatment for bacterial leaf spot.",
#     "Construct narrow Trenches alongside your tomato plant to ensure that water enters the plant’s root.",
#      "No Need To Do any thing",
#      "Strategies for managing late blight in tomato include planting resistant cultivars, eliminating volunteers spacing plants to increase airflow and reduce humidity, and applying preventive and effective fungicides to avoid infection.",
#      "Remove and destroy all crop debris post-harvest. Sanitize the greenhouse between crop seasons. Use fans and avoid overhead watering to minimize leaf wetness.",
#      "Fungicides containing either copper or potassium bicarbonate will help prevent the spreading of the disease.",
#      "In the intricate symphony of gardening, where tomato plants sway to the rhythms of the seasons, the presence of red spider mites can strike a discordant note. "
# ]


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
        return response_from_prediction(pred_arr,potato_classes,file_names,solutions=potato_solutions)
    
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
        return response_from_prediction(pred_arr,pepper_bell_classes,file_names,solutions=pepper_bell_solutions)
    
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
        return response_from_prediction(pred_arr,tomato_classes,file_names,solutions=tomato_solutions)
    
    except Exception as e:
        return {
            "error":str(e)
        }




if __name__=='__main__':
    uvicorn.run(app)