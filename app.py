from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from flask import Flask, url_for, request, render_template, redirect
app = Flask(__name__)

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()



@app.route("/")
def mainpage():
    return render_template("mainpage.html", upload = url_for("upload"))
@app.route("/upload")
def upload():
    return render_template("uploadphoto.html", results = url_for("results"))

@app.route("/results", methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        data = request.form['image']
        encoded_data = data.split(',')[1]
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Make the image a numpy array and reshape it to the models input shape.
        cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        print(image.shape[0])
        print(image.shape[1])
        
        #image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, -1)

        # Normalize the image array
        '''image = (image / 127.5) - 1'''

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        #image = data['image']
        #print(image)
        print("hi")
        
        
        return render_template("results.html")
    else:
        return render_template("results.html")

