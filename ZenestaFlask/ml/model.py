import json

from PIL import Image as pil
from keras.applications import inception_v3
import numpy as np

from keras import Model
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
from keras.utils.image_utils import img_to_array

from numpy._typing import NDArray
from werkzeug.datastructures import FileStorage

class PredictionModel(object):
    """description of class"""

    @staticmethod
    def create_pretrained():
        temp_model: Model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(255, 255, 3)))
        temp_model.save('./ml/model/pretrained.h5')

    model: Model

    def __init__(self, model_name: str):
        self.model = load_model(f'./ml/model/{model_name}')

    def predict(self, image_file: FileStorage) -> list:
        prediction: any = self.model.predict(self.process_image(image_file))

        # Since we send only one batch in
        prediction_decoded: any = inception_v3.decode_predictions(prediction)[0]

        #print(json.dumps(prediction_decoded))
        #print(prediction_decoded)

        return prediction_decoded

    def process_image(self, image_file: FileStorage) -> NDArray[any]:
        image = pil.open(image_file)
        image = image.convert('RGB')
        image = image.resize((255, 255))

        image_array: NDArray[any] = img_to_array(image) / 255
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
