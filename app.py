from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
from keras.api.models import load_model
from keras.api.preprocessing import image
import numpy as np

app = Flask(__name__)

#Load your trained model
model = load_model('AhmedCNN.keras')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        image_path = f'temp/{uploaded_file.filename}'  # Save the uploaded file temporarily
        uploaded_file.save(image_path)

        processed_image = preprocess_image(image_path)

#Make a prediction using your loaded model
#Replace this part with your actual prediction logic using the model
        prediction = model.predict(processed_image)

#For demonstration, assuming binary classification (0 or 1)
#Modify this based on your model's output format
        result = 'No_DR' if prediction[0][0] < 0.5 else 'DR'

        return jsonify({'prediction': result})

    return jsonify({'error': 'No file uploaded'})

if __name__ == '_main_':
    app.run(debug=True)