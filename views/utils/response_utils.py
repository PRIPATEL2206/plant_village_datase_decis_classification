import numpy as np

from .image_utils import image_to_nparry


def response_from_prediction(pred_arr,classes,files_name):
    return [ {
        "file_name":files_name[i],
        "class":classes[np.argmax(preds)],
        "confidence":round(float(preds.max()),2),
        "all":[
            {
                "class":classes[j],
                "confidence":round(float(pred)*100,2)
            }for j,pred in sorted(enumerate(preds), key=lambda x: x[1],reverse=True)
        ]
        } for i,preds in enumerate(pred_arr)]


async def files_to_np_array(files):
    return np.array([image_to_nparry(await file.read()) for file in files ])