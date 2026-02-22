import gradio as gr
import pickle
import numpy as np
import tempfile
import os
from extract_feature import extract_feature

with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

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

# Function for actual prediction
def classify_image(image):
    if image is None:
        return "No image uploaded", "", {}

    # Save PIL image to a temp file because extract_feature expects a path
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    features = extract_feature(tmp_path)
    os.remove(tmp_path)  # clean up temp file

    if features is None:
        return "Could not process image", "", {}

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probs = model.predict_proba(features_scaled)[0]

    fog_prob = probs[1]
    clear_prob = probs[0]
    if prediction == 1:
        label = "Foggy"
    else:
        label = "Clear"
    intensity = get_fog_intensity(fog_prob)

    return (
        label,
        intensity,
        {
            "Clear": round(float(clear_prob), 3),
            "Foggy": round(float(fog_prob), 3)
        }
    )

# gradio UI
with gr.Blocks(title="Road visibility Classifier") as app:
    gr.Markdown("Upload a road image and the model will classify "
                "fog or clear and estimate fog intensity.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Road Image")
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column():
            label_out = gr.Textbox(label="Classification")
            intensity_out = gr.Textbox(label="Fog Intensity")
            confidence_out = gr.Label(label="Confidence Scores")

    submit_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[label_out, intensity_out, confidence_out]
    )

app.launch()