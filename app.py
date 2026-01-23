from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the model you trained in cnn_model.py
MODEL_PATH = 'model.h5'
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Error: model.h5 not found. Run cnn_model.py first!")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # 1. Image Preprocessing with OpenCV
            # Convert file stream to OpenCV image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Resize to 32x32 (CIFAR-10 requirement)
            img_resized = cv2.resize(img, (32, 32))
            
            # Normalization [cite: 27]
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension (1, 32, 32, 3)
            img_final = np.expand_dims(img_normalized, axis=0)

            # 2. Model Prediction [cite: 41]
            preds = model.predict(img_final)
            score = tf.nn.softmax(preds[0]) # Get probabilities
            prediction_result = class_names[np.argmax(score)]

    return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
