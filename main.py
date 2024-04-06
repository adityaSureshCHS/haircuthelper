import cv2
import numpy as np
from flask import Flask, url_for, request, render_template, redirect

app = Flask(__name__)

@app.route("/")
def mainpage():
    image = cv2.imread('images/cut5.jpeg', 0)
    cv2.imshow("text", image)
    return render_template("mainpage.html")



app.run();