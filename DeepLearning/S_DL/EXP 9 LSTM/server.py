from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE, static_url_path='')
CORS(app)

# Global variables
model = None
tokenizer = None
max_sequence_length = 50

def load_model_and_tokenizer():
    global model, tokenizer, max_sequence_length
    
    try:
        # Load the H5 model
        model = load_model(os.path.join(BASE, 'LSTM_model.h5'))
        print("Model loaded successfully!")
        
        # Try to load tokenizer if it exists
        for fname in ('tokenizer.pickle', 'tokenizer.pkl'):
            tokenizer_file = os.path.join(BASE, fname)
            if os.path.exists(tokenizer_file):
                with open(tokenizer_file, 'rb') as f:
                    tokenizer = pickle.load(f)
                print(f"Tokenizer loaded from {fname}!")
                break
        else:
            print("Warning: tokenizer file not found.")
            tokenizer = None
        
        # Get max sequence length from model input shape
        max_sequence_length = model.input_shape[1]
        print(f"Max sequence length: {max_sequence_length}")
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_next_words(text, num_words):
    if model is None or tokenizer is None:
        return None, "Model or tokenizer not loaded"
    
    try:
        predictions = []
        current_text = text
        
        for _ in range(num_words):
            # Tokenize the current text
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            
            # Pad the sequence
            token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
            
            # Predict
            predicted = model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted, axis=-1)[0]
            
            # Get the word from index
            predicted_word = None
            for word, index in tokenizer.word_index.items():
                if index == predicted_word_index:
                    predicted_word = word
                    break
            
            if predicted_word is None:
                predicted_word = "[unknown]"
            
            predictions.append(predicted_word)
            current_text += " " + predicted_word
        
        return predictions, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return send_from_directory(BASE, 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'max_sequence_length': max_sequence_length
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    num_words = data.get('num_words', 5)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model or tokenizer not loaded'}), 500
    
    predictions, error = predict_next_words(text, num_words)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'predictions': predictions,
        'original_text': text
    })

if __name__ == '__main__':
    print("Loading model and tokenizer...")
    if load_model_and_tokenizer():
        print("Starting Flask server...")
        app.run(debug=True, port=5000)
    else:
        print("Failed to load model. Please check if LSTM_model.h5 exists.")
