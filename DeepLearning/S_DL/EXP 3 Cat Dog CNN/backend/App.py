from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)

# -------------------------
# Configuration
# -------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "..", "static", "uploads")
IMG_SIZE = (224, 224)   # MobileNet expects 224x224
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# Load trained model
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dog_MobileNet.h5")
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("cat_dog_MobileNet.h5 loaded successfully!")
    print(f"Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")

CLASS_NAMES = ["Cat", "Dog"]  # must match training order

# -------------------------
# Helpers
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)   # MobileNet normalization
    return img_array

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("Index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess image
    img_array = preprocess_image(file_path)

    # Predict — handle Keras model and sklearn/pickle model
    try:
        preds = model.predict(img_array)
        preds = np.array(preds)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Handle sigmoid (1 output), softmax (2 outputs), or direct class label
    if preds.ndim == 2 and preds.shape[-1] == 1:
        confidence = float(preds[0][0])
        label = "Dog" if confidence > 0.5 else "Cat"
        confidence = confidence if label == "Dog" else 1 - confidence
    elif preds.ndim == 2 and preds.shape[-1] >= 2:
        idx = int(np.argmax(preds[0]))
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        confidence = float(preds[0][idx])
    else:
        # Direct class prediction
        idx = int(preds.flatten()[0])
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        confidence = 1.0

    return jsonify({
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "image_url": "/" + file_path.replace("\\", "/")
    })

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
