import numpy as np
import pickle
from extract_feature import extract_feature

def predict(image_path):
    with open("model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

    features = extract_feature(image_path)
    if features is None:
        print("Could not read image.")
        return

    features_scaled = scaler.transform([features])

    prediction = model.predict(features_scaled)[0]
    fog_prob = model.predict_proba(features_scaled)[0][1]  # probability of foggy class
    intensity = get_fog_intensity(fog_prob)

    if predict == 1:
        label = "Foggy"
    else:
        label = "Clear"

    print(f"Classification : {label}")
    print(f"Fog Probability: {fog_prob:.2f}")
    print(f"Fog Intensity  : {intensity}")

def get_fog_intensity(fog_probability):
    # Convert fog probability to a human-readable intensity label.
    if fog_probability < 0.3:
        return "None"
    elif fog_probability < 0.5:
        return "Light"
    elif fog_probability < 0.7:
        return "Moderate"
    else:
        return "Heavy"