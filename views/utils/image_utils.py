from io import BytesIO
from  PIL import Image
import numpy as np


def image_to_nparry(img):
    img=Image.open(BytesIO(img)).convert("RGB").resize((256,256))
    return np.array(img)
