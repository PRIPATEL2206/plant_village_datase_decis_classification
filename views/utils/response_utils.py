import numpy as np

from .image_utils import image_to_nparry


def response_from_prediction(pred_arr,classes,files,solutions):
    # print(solutions[np.argmax(pred_arr[0])])
    file_names=[i.filename for i in files]
    ans= [ {
        "file_name":file_names[i],
        "class":classes[np.argmax(preds)],
        "solution":solutions[np.argmax(preds)],
        "confidence":round(float(preds.max()),4),
        "all":[
            {
                "class":classes[j],
                "confidence":round(float(pred)*100,2)
            }for j,pred in sorted(enumerate(preds), key=lambda x: x[1],reverse=True)
        ]
        } for i,preds in enumerate(pred_arr)]
    return ans


async def files_to_np_array(files):
    return np.array([image_to_nparry(await file.read()) for file in files ])