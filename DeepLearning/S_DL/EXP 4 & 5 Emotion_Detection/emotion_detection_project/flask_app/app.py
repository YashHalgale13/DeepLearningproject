from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import base64
import os

app = Flask(__name__)

# Load emotion detection model
emotion_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "emotion_model.h5")
emotion_model = tf.keras.models.load_model(emotion_model_path)

EMOTION_CATEGORIES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

@app.route("/", methods=["GET", "POST"])
def index():
    emotion_prediction = ""
    emotion_img_base64 = None
    
    if request.method == "POST":
        if "emotion_file" in request.files:
            file = request.files.get("emotion_file")
            
            if file and file.filename:
                file_content = file.read()
                file_bytes = np.frombuffer(file_content, np.uint8)
                
                # Convert image to base64 for display
                emotion_img_base64 = base64.b64encode(file_content).decode('utf-8')
                
                # Process image
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    img = img.reshape(1, 48, 48, 1) / 255.0
                    
                    preds = emotion_model.predict(img)
                    emotion_prediction = EMOTION_CATEGORIES[int(np.argmax(preds))]
    
    return render_template(
        "index.html",
        emotion_prediction=emotion_prediction,
        emotion_image_data=emotion_img_base64
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)