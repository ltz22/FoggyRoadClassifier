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

    # Feature 1: brightness mean (foggy = high)
    brightness_mean = np.mean(hsv[:, :, 2])

    #Feature 2: brightness standard deviation (foggy = uniform brightness)
    brightness_std = np.std(hsv[:, :, 2])

    # Feature 3: contrast (foggy = low contrast)
    contrast = gray.std()

    # Feature 4: saturation mean (foggy images have low saturation)
    saturation_mean = np.mean(hsv[:, :, 1])

    # Feature 5: dark channel (foggy images have high dark channel)
    min_channel = np.min(img_resized, axis=2)
    dark_channel = np.mean(min_channel)

    # Feature 6: edge density (foggy images have fewer sharp edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    return [brightness_mean, brightness_std, contrast,
            saturation_mean, dark_channel, edge_density]

def load_dataset(data_dir):
    X, y = [], []
    class_map = {"clear": 0, "foggy": 1}

    for label_name, label_val in class_map.items():
        folder = os.path.join(data_dir, label_name)
        for filename in os.listdir(folder):
            if filename.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(folder, filename)
                features = extract_feature(path)
                if features:
                    X.append(features)
                    y.append(label_val)

    return np.array(X), np.array(y)