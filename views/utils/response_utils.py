import numpy as np

from .image_utils import image_to_nparry


def response_from_prediction(pred_arr,classes):
    return [ {
        "class":classes[np.argmax(pred)],
        "confidence":float(pred.max())
        } for pred in pred_arr]


async def file_to_np_array(files):
    return np.array([image_to_nparry(await file.read()) for file in files ])