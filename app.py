from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

model = tf.keras.models.load_model("coffee_leaf_model.h5")
categories = np.load("categories.npy")

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

disease_info = {
    "Healthy": {
        "description": "The coffee leaf is healthy with no signs of infection or infestation.",
        "cause": "N/A",
        "cure": "No treatment necessary. Keep the plant under proper care and monitor for future symptoms."
    },
    "Miner": {
        "description": "Leaf miners are larvae that feed between the upper and lower surfaces of leaves, creating visible tunnels.",
        "cause": "Caused by the larvae of various insect species (typically moths or flies) laying eggs on the leaves.",
        "cure": "Remove affected leaves. Use neem oil sprays or introduce natural predators like parasitic wasps."
    },
    "Phoma": {
        "description": "Phoma leaf spot is a fungal disease that creates brown or black lesions on leaves, leading to defoliation.",
        "cause": "Spread by fungal spores in wet, humid conditions — especially when leaves remain wet for long.",
        "cure": "Apply fungicides like mancozeb or chlorothalonil. Improve air circulation and avoid overhead watering."
    },
    "Red Spider Mite": {
        "description": "Tiny red mites that suck sap from leaves, causing yellowing, speckling, and eventual leaf drop.",
        "cause": "Dry, hot conditions that support mite population growth; often due to lack of humidity.",
        "cure": "Increase humidity, rinse leaves with water, use miticides or insecticidal soap. Remove heavily infested leaves."
    },
    "Rust": {
        "description": "A common fungal disease causing yellow-orange powdery spots on the underside of leaves.",
        "cause": "Caused by the Hemileia vastatrix fungus; thrives in moist, shaded environments.",
        "cure": "Use copper-based or systemic fungicides. Remove and destroy infected leaves. Ensure proper spacing and sunlight."
    }
}



def predict_image(img_path):
    img_size = 128
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    return categories[class_idx]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded.jpg')
        file.save(filepath)
        result = predict_image(filepath)
        info = disease_info.get(result, {
            "description": "No information available.",
            "cause": "Unknown",
            "cure": "No recommended treatment."
        })
        return render_template("result.html", result=result, image_file='uploaded.jpg', info=info)

if __name__ == "__main__":
    app.run(debug=True)
