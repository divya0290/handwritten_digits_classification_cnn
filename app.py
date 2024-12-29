# from flask import Flask, request, jsonify, render_template
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import os
# app = Flask(__name__)
# try:
#     # Try loading with custom_objects
#     model = tf.keras.models.load_model('mnist_cnn_model.h5', compile=False)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     # Alternative loading approach
#     try:
#         model = tf.keras.models.load_model('mnist_cnn_model.h5', 
#                                          custom_objects={'InputLayer': tf.keras.layers.InputLayer},
#                                          compile=False)
#     except Exception as e:
#         print(f"Second loading attempt failed: {e}")
#         raise

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         img = Image.open(file.stream).convert('L')  # Convert to grayscale
#         img = img.resize((28, 28))                 # Resize to 28x28
#         img = np.array(img) / 255.0                # Normalize pixel values
#         img = img.reshape(1, 28, 28, 1)            # Reshape to match model input

#         prediction = model.predict(img)
#         digit = np.argmax(prediction)

#         return jsonify({'digit': int(digit)})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the pre-trained model
try:
    model = tf.keras.models.load_model('mnist_model.h5')
    logging.info("Loaded pre-trained model successfully")
except Exception as e:
    logging.error(f"Error loading pre-trained model: {e}")
    # Create a basic model as fallback
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    logging.warning("Created fallback model")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.error("No file selected")
            return jsonify({'error': 'No file selected', 'success': False}), 400

        # Read and preprocess the image
        img = Image.open(file).convert('L')
        logging.info(f"Image opened, size: {img.size}")
        
        img = img.resize((28, 28))
        logging.info("Image resized to 28x28")
        
        img_array = np.array(img)
        logging.info(f"Image converted to array, shape: {img_array.shape}")
        
        img_array = img_array.reshape(1, 28, 28, 1)
        img_array = img_array / 255.0
        logging.info(f"Final input shape: {img_array.shape}")
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_digit])
        
        logging.info(f"Prediction made: {predicted_digit} with confidence: {confidence}")
        
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': float(confidence),
            'success': True
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'success': True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)