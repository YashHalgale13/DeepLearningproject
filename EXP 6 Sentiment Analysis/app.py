"""
EXP 6 — RNN for Text Sentiment Analysis (Positive / Negative)
Uses imdb_rnn_model (1).h5 trained on the IMDB dataset.
"""

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np
import os
import re

app = Flask(__name__)

BASE       = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, 'imdb_rnn_model (1).h5')

# ── load model ──
model = None
try:
    model = load_model(MODEL_PATH, compile=False)
    print(f"Model loaded! Input shape: {model.input_shape}")
except Exception as e:
    print(f"Error loading model: {e}")

# ── IMDB config ──
VOCAB_SIZE = 10000
MAX_LEN    = 500

# ── load IMDB word index ──
print("Loading IMDB word index...")
word_index = None
try:
    word_index = imdb.get_word_index()
    print(f"Word index loaded — {len(word_index)} words")
except Exception as e:
    print(f"Could not download word index ({e}), using fallback.")
    word_index = {
        'the':1,'and':2,'a':3,'of':4,'to':5,'is':6,'in':7,'it':8,'i':9,
        'this':10,'that':11,'was':12,'as':13,'for':14,'with':15,'movie':16,
        'film':17,'but':18,'not':19,'you':20,'are':21,'on':22,'have':23,
        'be':24,'good':30,'great':31,'bad':32,'best':33,'worst':34,
        'love':35,'like':36,'really':37,'very':38,'excellent':39,'awful':40,
        'boring':41,'amazing':42,'terrible':43,'wonderful':44,'horrible':45,
        'fantastic':46,'poor':47,'brilliant':48,'dull':49,'superb':50
    }


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # IMDB word index reserves 0=pad, 1=start, 2=OOV, 3=unused → shift by +3
    seq = []
    for w in text.split():
        idx = word_index.get(w, 0)
        shifted = idx + 3
        if shifted < VOCAB_SIZE:
            seq.append(shifted)
        else:
            seq.append(2)  # OOV
    seq = seq if seq else [2]
    return pad_sequences([seq], maxlen=MAX_LEN, padding='pre').astype(np.int32)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data   = request.get_json()
    review = data.get('review', '').strip()
    if not review:
        return jsonify({'error': 'Please enter a review'}), 400

    try:
        processed = preprocess(review)
        score     = float(model.predict(processed, verbose=0).flatten()[0])
        score     = max(0.0, min(1.0, score))

        if score >= 0.5:
            sentiment   = 'POSITIVE'
            confidence  = round(score * 100, 2)
        else:
            sentiment   = 'NEGATIVE'
            confidence  = round((1 - score) * 100, 2)

        return jsonify({
            'sentiment':  sentiment,
            'confidence': confidence,
            'raw_score':  round(score, 4),
            'review':     review[:120] + '...' if len(review) > 120 else review
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})


if __name__ == '__main__':
    print("Sentiment Analysis → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
