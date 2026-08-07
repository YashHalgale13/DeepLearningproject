from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load GRU model
model = None
tokenizer = None
max_len = 100

model_path = os.path.join(os.path.dirname(__file__), "gru_model.keras")
tokenizer_path = os.path.join(os.path.dirname(__file__), "GRU_Tokenizer.pkl")

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded! Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading .keras model: {e}")
    try:
        h5_path = os.path.join(os.path.dirname(__file__), "GRU_Spam_Detector (1).h5")
        model = tf.keras.models.load_model(h5_path)
        print(f"Loaded H5 model. Input shape: {model.input_shape}")
    except Exception as e2:
        print(f"Error loading H5 model: {e2}")

try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded!")
    # Infer max_len from model input shape if possible
    if model is not None:
        max_len = model.input_shape[1]
        print(f"Max sequence length: {max_len}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

def preprocess_sms(text):
    """Clean and tokenize SMS text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    seq = tokenizer.texts_to_sequences([text])
    # Use 'pre' padding — GRU/LSTM models expect zeros at the start
    padded = pad_sequences(seq, maxlen=max_len, padding='pre', truncating='pre')
    return padded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500

    data = request.get_json()
    sms_text = data.get('sms', '').strip()

    if not sms_text:
        return jsonify({'error': 'Please enter an SMS message'}), 400

    try:
        processed = preprocess_sms(sms_text)
        prediction = model.predict(processed, verbose=0)
        score = float(prediction.flatten()[0])
        print(f"Raw score: {score:.4f}")  # debug — check terminal

        if score >= 0.5:
            label = "SPAM"
            confidence = round(score * 100, 2)
        else:
            label = "HAM (Not Spam)"
            confidence = round((1 - score) * 100, 2)

        return jsonify({
            'label': label,
            'confidence': confidence,
            'raw_score': round(score, 4),
            'sms': sms_text[:120] + '...' if len(sms_text) > 120 else sms_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

if __name__ == '__main__':
    print("SMS Spam Detector starting at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
