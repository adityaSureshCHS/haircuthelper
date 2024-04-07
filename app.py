from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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
        return render_template("results.html")
    else:
        return render_template("results.html")

