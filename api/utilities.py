import keras
import numpy as np
import requests
from PIL import Image
from io import BytesIO

model = keras.models.load_model('./model')
class_names = ['Pepper Bell_Bacterial Spot', 'Pepper Bell_Healthy', 'Potato_Early Blight', 'Potato_Healthy', 'Potato_Late Blight', 'Tomato_Target Spot', 'Tomato_Mosaic virus', 'Tomato_YellowLeaf Curl Virus', 'Tomato_Bacterial Spot', 'Tomato_Early Blight', 'Tomato_Healthy', 'Tomato_Late Blight', 'Tomato_Leaf Mold', 'Tomato_Septoria Leaf Spot', 'Tomato_Two Spotted Spider Mite']

def predict(url):
    img = load_img(url)
    return predict_img(img)

def load_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224,224))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img;

def predict_img(img):
    output = model.predict(img)
    ind = np.argsort(output[0])
    ind = ind[-5 :]
    results = []
    for index, val in enumerate(ind):
        obj = {}
        obj['disease'] = class_names[val]
        obj['propability'] = float(output[0][val])
        results.append(obj)
    return results