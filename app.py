import cv2
import numpy as np
from flask import Flask, url_for, request, render_template, redirect

app = Flask(__name__)

@app.route("/")
def mainpage():
    return render_template("mainpage.html", upload = url_for("upload"))

@app.route("/upload")
def upload():
    return render_template("uploadphoto.html", results = url_for("results"))

@app.route("/results")
def results():
    return render_template("results.html")


def display_image():
    image = cv2.imread('images/cut5.jpeg', 0)
    cv2.imshow("text", image)
    cv2.waitKey(5)
    cv2.destroyAllWindows()

