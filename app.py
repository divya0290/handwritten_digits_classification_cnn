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



app = Flask(__name__)

# Define the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile the model
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Load the weights from your existing model
try:
    model.load_weights('mnist_cnn_model.h5')
except Exception as e:
    print(f"Error loading weights: {e}")
    # If loading fails, we'll use the untrained model
    pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    # Read and preprocess the image
    img = Image.open(file).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction[0])
    
    return jsonify({'prediction': int(predicted_digit)}) 

if __name__ == "__main__":
      app.run(debug=True)