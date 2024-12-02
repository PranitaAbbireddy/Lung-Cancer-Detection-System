# app.py
from flask import Flask, render_template, request, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'model/lung_cancer_detector.h5'
model = load_model(MODEL_PATH)

def preprocess_image(img_path):
    IMG_SIZE = 128
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)[0][0]
        result = "Lung Cancer Detected" if prediction > 0.5 else "No Lung Cancer Detected"
        return render_template('index.html', result=result, file_path=file_path)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)

