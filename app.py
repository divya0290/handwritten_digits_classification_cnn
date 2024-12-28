from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image


app = Flask(__name__)


model = tf.keras.models.load_model('mnist_cnn_model.h5')

@app.route('/')
def home():
    return "Handwritten Digit Classifier API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
       
        img = Image.open(file.stream).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))                 # Resize to 28x28
        img = np.array(img) / 255               # Normalize pixel values
        img = img.reshape(1, 28, 28, 1)            # Reshape to match model input

       
        prediction = model.predict(img)
        digit = np.argmax(prediction)

        return jsonify({'digit': int(digit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
