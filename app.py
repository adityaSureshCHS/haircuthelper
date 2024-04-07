import requests
import json
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
import os
import base64
from flask import Flask, url_for, request, render_template, redirect
app = Flask(__name__)
client = OpenAI(
    api_key=os.environ.get("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")
)
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

#hi
arayOfMen =[
    ["Crew Cut", "middle Part", "Textured Crop", "Slicked Back", "Quiff", "Pompadour"],  # Oval
    ["Crew Cut", "Buzz Cut", "Side Part", "Faux Hawk", "Spiky Hair with a Mid Fade", "High Fade with a Pompadour"],  # Round
    ["Crew Cut", "Buzz Cut", "Brushed Back Hair", "Long Hair with Layers", "Textured Quiff", "Side Part Pompadour"],  # Rectangular
    ["Crew Cut", "Buzz Cut", "Undercut", "Textured Crop", "The Ivy League", "Classic Pompadour"]  # Square
]
arrayOfWomen =[
    ["Long Layers", "Bob Cut", "Pixie Cut", "Soft Fringe", "Side-Swept Bangs", "Wavy Textured Lob"],  # Oval
    ["Long Straight Hair with Layers", "High Volume Waves", "Tousled Lob", "Choppy Pixie", "Asymmetrical Bob", "Deep Side Part"],  # Round
    ["Mid-Length With Volume", "Soft Layers with Bangs", "Long Bob (Lob)", "Curtain Bangs", "Wavy Shag Cut", "U-Cut with Waves"],  # Rectangular
    ["Long Straight Hair", "Wavy Lob with Bangs", "Shoulder-Length Bob", "Soft Waves with Side-Swept Bangs", "Angled Bob", "Layered Pixie"]  # Square
]
@app.route("/")
def mainpage():
    return render_template("mainpage.html", upload = url_for("upload"))
@app.route("/upload")
def upload():
    return render_template("uploadphoto.html", results = url_for("results"))

@app.route("/results", methods=["POST", "GET"])
def results():
    if request.method == 'POST':
        global data
        data = request.form['image']
        global encoded_data 
        encoded_data = data.split(',')[1]
        global nparr 
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        global image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Make the image a numpy array and reshape it to the models input shape.
        image = cv2.resize(image, (224, 224))
        print(image.shape[0])
        print(image.shape[1])
        
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, -1)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        global prediction
        prediction = model.predict(image)
        global index
        index = np.argmax(prediction)
        global class_name
        class_name= class_names[index]
        global confidence_score 
        onfidence_score = prediction[0][index]
        global haircuts 
        haircuts = whatType(class_name, confidence_score*100)

        # Print prediction and confidence score
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    return redirect(url_for("display_analysis"))
@app.route("/display_analysis")
def display_analysis():
    final_data = []
    for i in haircuts:
        cv2.imwrite("displayedimage1.png", image)
        response = client.images.edit(
            model="dall-e-3",
            image=open("displayedimage1.png", "rb"),
            mask=open("images-removebg-preview.png", "rb"),
            prompt="Keep the entire base image except change the haircut to a " + haircuts[i],
            n=1
        )
        image_url = response.data[0].url
        final_data.append(image_url)
    
    render_template("results.html", final_data = final_data, haircuts=haircuts)

def whatType(confScore, percent): 

    if(confScore =="Oval"): 
        if(percent> 26 and percent <=50):
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[0,5]
            else: 
                womenPlusMen[0] = arrayOfMen[0,4] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[0,5]
            else: 
                womenPlusMen[1] = arrayOfWomen[0,4] 
            return womenPlusMen
        if(percent >50 and percent<=75): 
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[0,3]
            else: 
                womenPlusMen[0] = arrayOfMen[0,2] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[0,3]
            else: 
                womenPlusMen[1] = arrayOfWomen[0,2] 
            return womenPlusMen
        if(percent >75):
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[0,1]
            else: 
                womenPlusMen[0] = arrayOfMen[0,0] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[0,1]
            else: 
                womenPlusMen[1] = arrayOfWomen[0,0] 
            return womenPlusMen
    if(confScore == "Round"): 
        if(percent> 26 and percent <=50):
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[1,5]
            else: 
                womenPlusMen[0] = arrayOfMen[1,4] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[1,5]
            else: 
                womenPlusMen[1] = arrayOfWomen[1,4] 
            return womenPlusMen
        if(percent >50 and percent<=75): 
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[1,3]
            else: 
                womenPlusMen[0] = arrayOfMen[1,2] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[1,3]
            else: 
                womenPlusMen[1] = arrayOfWomen[1,2] 
            return womenPlusMen
        if(percent >75):
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[1,1]
            else: 
                womenPlusMen[0] = arrayOfMen[1,0] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[1,1]
            else: 
                womenPlusMen[1] = arrayOfWomen[1,0] 
            return womenPlusMen
    if(confScore == "Rectangular"):
        if(percent> 26 and percent <=50):
            if(percent> 26 and percent <=50):
                womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[2,5]
            else: 
                womenPlusMen[0] = arrayOfMen[2,4] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[2,5]
            else: 
                womenPlusMen[1] = arrayOfWomen[2,4] 
            return womenPlusMen
        if(percent >50 and percent<=75): 
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[2,3]
            else: 
                womenPlusMen[0] = arrayOfMen[2,2] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[2,3]
            else: 
                womenPlusMen[1] = arrayOfWomen[2,2] 
            return womenPlusMen
        if(percent >75):
            womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[2,1]
            else: 
                womenPlusMen[0] = arrayOfMen[2,0] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[2,1]
            else: 
                womenPlusMen[1] = arrayOfWomen[2,0] 
            return womenPlusMen
    if(confScore == "Square"):
        if(percent> 26 and percent <=50):
            if(percent> 26 and percent <=50):
                womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[3,5]
            else: 
                womenPlusMen[0] = arrayOfMen[3,4] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[3,5]
            else: 
                womenPlusMen[1] = arrayOfWomen[3,4] 
            return womenPlusMen
        if(percent >50 and percent<=75): 
            if(percent> 26 and percent <=50):
                womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[3,3]
            else: 
                womenPlusMen[0] = arrayOfMen[3,2] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[3,3]
            else: 
                womenPlusMen[1] = arrayOfWomen[3,2] 
            return womenPlusMen 
        if(percent >75):
            if(percent> 26 and percent <=50):
                womenPlusMen = [0, 0] 
            if(math.random<=0.5):
                    womenPlusMen[0] = arrayOfMen[3,1]
            else: 
                womenPlusMen[0] = arrayOfMen[3,1] 
            if(math.random<=0.5):
                    womenPlusMen[1] = arrayOfWomen[3,2]
            else: 
                womenPlusMen[1] = arrayOfWomen[3,1] 
            return womenPlusMen 

    
