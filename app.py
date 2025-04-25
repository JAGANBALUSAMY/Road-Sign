from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("gtsrb_model.h5")

# Load class labels from JSON file
with open("class_label.json", "r") as file:
    class_labels = json.load(file)

# Define image preprocessing function
def preprocess_image(image_path):
    IMG_HEIGHT, IMG_WIDTH = 64, 64  # Match training size
    img = Image.open(image_path).convert("RGB")  # Convert PNG to RGB (if needed)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match training image size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension for prediction
    return img

# Define prediction function
def predict_road_sign(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)  # Predict using the trained model
    predicted_class = np.argmax(predictions)  # Get the predicted class ID
    return class_labels[str(predicted_class)]  # Get the human-readable label

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the upload page

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    
    file = request.files['image']  # Get the uploaded file

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded image to a temporary file
    temp_image_path = os.path.join('uploads', file.filename)
    file.save(temp_image_path)
    
    # Predict the road sign
    try:
        result = predict_road_sign(temp_image_path)
        return jsonify({'prediction': result}), 200  # Return the result as JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
