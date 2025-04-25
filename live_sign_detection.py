import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("gtsrb_model.h5")

# Load class labels from JSON file
with open("class_label.json", "r") as file:
    class_labels = json.load(file)

# Define confidence threshold (Adjustable)
CONFIDENCE_THRESHOLD = 0.7  # 70% confidence required

# Define image preprocessing function
def preprocess_image(frame):
    IMG_HEIGHT, IMG_WIDTH = 64, 64  # Match training size
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = Image.fromarray(img)  # Convert to PIL image
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize to match training images
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict road sign from frame
def predict_road_sign(frame):
    img = preprocess_image(frame)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)  # Get highest probability class ID
    confidence = np.max(predictions)  # Get confidence score

    if confidence >= CONFIDENCE_THRESHOLD:
        return class_labels[str(predicted_class)], confidence
    else:
        return "No sign detected", confidence  # Return nothing if confidence is low

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no frame is captured

        # Predict the road sign in the frame
        result, confidence = predict_road_sign(frame)

        # Convert BGR to RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display image using Matplotlib
        plt.imshow(frame_rgb)
        plt.title(f"Prediction: {result} ({confidence:.2f})")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()  # Clear the previous frame

        # Check if 'q' is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing webcam...")
            break

except KeyboardInterrupt:
    print("Interrupted by user, closing webcam...")

# Release resources properly
cap.release()
cv2.destroyAllWindows()
plt.close('all')
