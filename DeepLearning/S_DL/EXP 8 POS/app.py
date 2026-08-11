"""
EXP 8 — Synchronous Many-to-Many RNN for POS Tagging
Architecture: Embedding -> Bidirectional LSTM -> TimeDistributed Dense
Trained on Penn Treebank (NLTK). Auto-trains on first run (~1-2 min).
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os, pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE, 'pos_model.h5')
WORD_IDX_PATH = os.path.join(BASE, 'word_index.pkl')
TAG_IDX_PATH  = os.path.join(BASE, 'tag_index.pkl')

MAX_LEN    = 50
EMBED_DIM  = 128
LSTM_UNITS = 256

app = Flask(__name__)

# ── Penn Treebank tag descriptions ──
TAG_DESC = {
    'CC':'Coordinating conjunction', 'CD':'Cardinal number', 'DT':'Determiner',
    'EX':'Existential there', 'FW':'Foreign word', 'IN':'Preposition / subord. conj.',
    'JJ':'Adjective', 'JJR':'Adjective, comparative', 'JJS':'Adjective, superlative',
    'LS':'List item marker', 'MD':'Modal verb', 'NN':'Noun, singular',
    'NNS':'Noun, plural', 'NNP':'Proper noun, singular', 'NNPS':'Proper noun, plural',
    'PDT':'Predeterminer', 'POS':'Possessive ending', 'PRP':'Personal pronoun',
    'PRP$':'Possessive pronoun', 'RB':'Adverb', 'RBR':'Adverb, comparative',
    'RBS':'Adverb, superlative', 'RP':'Particle', 'SYM':'Symbol', 'TO':'to',
    'UH':'Interjection', 'VB':'Verb, base form', 'VBD':'Verb, past tense',
    'VBG':'Verb, gerund / present participle', 'VBN':'Verb, past participle',
    'VBP':'Verb, non-3rd person singular', 'VBZ':'Verb, 3rd person singular',
    'WDT':'Wh-determiner', 'WP':'Wh-pronoun', 'WP$':'Possessive wh-pronoun',
    'WRB':'Wh-adverb', '.':'Punctuation', ',':'Comma', ':':'Colon / semicolon',
    '``':'Open quote', "''":"Close quote", '-LRB-':'Left bracket',
    '-RRB-':'Right bracket', '$':'Dollar sign', '#':'Pound sign',
}

model      = None
word_index = {}
tag_index  = {}
idx_to_tag = {}


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def build_and_train():
    import nltk
    print("Downloading NLTK data...")
    nltk.download('treebank',                   quiet=True)
    nltk.download('conll2000',                  quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    from nltk.corpus import treebank, conll2000

    # Combine treebank + conll2000 for more training data
    sentences = list(treebank.tagged_sents())
    try:
        sentences += list(conll2000.tagged_sents())
    except Exception:
        pass
    print(f"Total training sentences: {len(sentences)}")

    # Build vocabularies
    words, tags = set(), set()
    for sent in sentences:
        for w, t in sent:
            words.add(w.lower())
            tags.add(t)

    w_idx = {w: i + 2 for i, w in enumerate(sorted(words))}
    w_idx['<PAD>'] = 0
    w_idx['<UNK>'] = 1

    t_idx = {t: i + 1 for i, t in enumerate(sorted(tags))}
    t_idx['<PAD>'] = 0

    # Encode sequences
    X, y = [], []
    for sent in sentences:
        xs = [w_idx.get(w.lower(), 1) for w, _ in sent]
        ys = [t_idx[t]               for _, t in sent]
        X.append(xs)
        y.append(ys)

    X     = pad_sequences(X, maxlen=MAX_LEN, padding='post', value=0)
    y     = pad_sequences(y, maxlen=MAX_LEN, padding='post', value=0)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(t_idx) + 1)

    vocab_size = len(w_idx) + 1
    num_tags   = len(t_idx) + 1

    m = Sequential([
        Embedding(vocab_size, EMBED_DIM, mask_zero=True),
        Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(LSTM_UNITS // 2, return_sequences=True)),
        Dropout(0.3),
        TimeDistributed(Dense(num_tags, activation='softmax'))
    ])
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print("Training POS model (this takes ~3-5 minutes)...")
    m.fit(X, y_cat, epochs=15, batch_size=32,
          validation_split=0.1, verbose=1)

    m.save(MODEL_PATH)
    with open(WORD_IDX_PATH, 'wb') as f: pickle.dump(w_idx, f)
    with open(TAG_IDX_PATH,  'wb') as f: pickle.dump(t_idx, f)
    print("Model + vocab saved.")
    return m, w_idx, t_idx


def load_or_train():
    global model, word_index, tag_index, idx_to_tag

    if (os.path.exists(MODEL_PATH) and
            os.path.exists(WORD_IDX_PATH) and
            os.path.exists(TAG_IDX_PATH)):
        try:
            model = load_model(MODEL_PATH)
            with open(WORD_IDX_PATH, 'rb') as f: word_index = pickle.load(f)
            with open(TAG_IDX_PATH,  'rb') as f: tag_index  = pickle.load(f)
            idx_to_tag = {v: k for k, v in tag_index.items()}
            print(f"POS model loaded. Tags: {len(tag_index)}, Words: {len(word_index)}")
            return
        except Exception as e:
            print(f"Load failed ({e}), retraining...")

    model, word_index, tag_index = build_and_train()
    idx_to_tag = {v: k for k, v in tag_index.items()}


load_or_train()


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def tag_sentence(sentence: str):
    tokens = sentence.strip().split()
    if not tokens:
        return []

    seq    = [word_index.get(w.lower(), 1) for w in tokens]
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding='post', value=0)
    preds  = model.predict(padded, verbose=0)[0]   # (MAX_LEN, num_tags)

    results = []
    for i, token in enumerate(tokens):
        tag_idx = int(np.argmax(preds[i]))
        tag     = idx_to_tag.get(tag_idx, 'NN')
        # skip PAD tag
        if tag == '<PAD>':
            tag = 'NN'
        results.append({
            'word':        token,
            'tag':         tag,
            'description': TAG_DESC.get(tag, tag),
            'confidence':  round(float(np.max(preds[i])) * 100, 1)
        })
    return results


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tag', methods=['POST'])
def tag():
    data     = request.get_json()
    sentence = data.get('sentence', '').strip()
    if not sentence:
        return jsonify({'error': 'Please enter a sentence'}), 400
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        tagged = tag_sentence(sentence)
        return jsonify({'tokens': tagged, 'count': len(tagged)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'model_loaded': model is not None,
        'vocab_size':   len(word_index),
        'num_tags':     len(tag_index)
    })


if __name__ == '__main__':
    print("POS Tagger ready -> http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
