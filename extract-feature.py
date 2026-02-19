import cv2
import numpy as np
import os

def extract_feature(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img_resized = cv2.resize(img, (224, 224))

    # Convert to grayscale and HSV
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)