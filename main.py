import cv2
import numpy as np
image = cv2.imread('cut6.jpg')
if image is not None:
    # Save the image
    cv2.imwrite('cut6.jpg', image)
    print("Image saved successfully.")
else:
    print("Error: Unable to load the image.")

