
"""
EXP 7 — One-to-Many RNN for Image Captioning
Encoder: VGG16 (pretrained ImageNet) extracts a 512-dim feature vector.
Decoder: Greedy LSTM decoder if a trained model + tokenizer are present.
Fallback: VGG16 activation-based scene description when no trained model exists.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import os, io, base64, pickle
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

BASE            = os.path.dirname(os.path.abspath(__file__))
MAX_LEN         = 34

# Trained model paths — set after running train_model.py
MODEL_PATH     = os.path.join(BASE, 'caption_model_trained.h5')
TOKENIZER_PATH = os.path.join(BASE, 'caption_tokenizer.pkl')

# ── VGG16 feature extractor ──
print("Loading VGG16...")
feature_extractor = VGG16(weights='imagenet', include_top=False, pooling='avg')
print("VGG16 ready.")

# ── optional caption model + tokenizer ──
caption_model = None
tokenizer     = None
idx_to_word   = {}

if os.path.exists(MODEL_PATH):
    try:
        caption_model = load_model(MODEL_PATH, compile=False)
        print(f"Caption model loaded. Inputs: {[i.shape for i in caption_model.inputs]}")
    except Exception as e:
        print(f"Caption model load error: {e}")

if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        idx_to_word = {v: k for k, v in tokenizer.word_index.items()}
        idx_to_word[0] = ''
        print(f"Tokenizer loaded — vocab: {len(tokenizer.word_index)}")
    except Exception as e:
        print(f"Tokenizer load error: {e}")


# ─────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────
def extract_features(pil_image):
    img = pil_image.resize((224, 224)).convert('RGB')
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return feature_extractor.predict(arr, verbose=0)   # (1, 512)


# ─────────────────────────────────────────────
# Greedy RNN decoder (needs trained model)
# ─────────────────────────────────────────────
def greedy_caption(features):
    in_text = 'startseq'
    for _ in range(MAX_LEN):
        seq  = tokenizer.texts_to_sequences([in_text])[0]
        seq  = pad_sequences([seq], maxlen=MAX_LEN)
        yhat = caption_model.predict([features, seq], verbose=0)
        widx = int(np.argmax(yhat))
        word = idx_to_word.get(widx, '')
        if not word or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq', '').strip()


def model_is_trained():
    """Detect random weights via output entropy."""
    if caption_model is None or tokenizer is None:
        return False
    try:
        dummy_feat = np.zeros((1, 512))
        dummy_seq  = np.zeros((1, MAX_LEN))
        out     = caption_model.predict([dummy_feat, dummy_seq], verbose=0)
        probs   = out[0]
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return entropy < 0.95 * np.log(len(probs))
    except Exception:
        return False


# ─────────────────────────────────────────────
# VGG16 activation-based scene description
# ─────────────────────────────────────────────

# ImageNet class groups mapped to scene words
SCENE_CONCEPTS = {
    'animal':   list(range(0,   400)),   # various animals
    'bird':     list(range(7,   24)),
    'dog':      list(range(151, 269)),
    'cat':      list(range(281, 294)),
    'vehicle':  list(range(400, 500)),
    'food':     list(range(924, 970)),
    'furniture':list(range(559, 600)),
    'person':   [878, 879, 880],
    'nature':   list(range(970, 1000)),
}

# Load lightweight MobileNetV2 for ImageNet class prediction
print("Loading MobileNetV2 for scene classification...")
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    scene_classifier = MobileNetV2(weights='imagenet', include_top=True)
    print("MobileNetV2 ready.")
except Exception as e:
    scene_classifier = None
    print(f"MobileNetV2 not available: {e}")


def classify_scene(pil_image):
    """Get top ImageNet predictions for scene understanding."""
    if scene_classifier is None:
        return [], []
    img = pil_image.resize((224, 224)).convert('RGB')
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = mn_preprocess(arr)
    preds = scene_classifier.predict(arr, verbose=0)
    top   = decode_predictions(preds, top=5)[0]   # [(id, label, score), ...]
    return top


def feature_based_caption(features, pil_image):
    """
    Build a natural caption using MobileNetV2 top-5 + color/brightness analysis.
    """
    top_preds = classify_scene(pil_image)
    objects   = [(label.replace('_', ' '), score) for (_, label, score) in top_preds if score > 0.03]

    # ── color analysis ──
    small  = np.array(pil_image.resize((64, 64)).convert('RGB')) / 255.0
    pixels = small.reshape(-1, 3)
    r, g, b = pixels[:,0].mean(), pixels[:,1].mean(), pixels[:,2].mean()
    bright  = pixels.mean()

    # ── detect if it's a portrait (face-like crop: tall, skin tones dominant) ──
    w, h   = pil_image.size
    ratio  = w / h
    # skin tone: high red, moderate green, low blue
    skin_mask = (pixels[:,0] > 0.4) & (pixels[:,0] > pixels[:,1]) & (pixels[:,1] > pixels[:,2])
    skin_ratio = skin_mask.mean()

    is_portrait = (ratio < 1.2) and (skin_ratio > 0.15)

    # ── clothing / accessory labels that hint at a person ──
    person_hints = {
        'suit', 'tie', 'bow tie', 'lab coat', 'jersey', 'shirt', 'jacket',
        'dress', 'gown', 'uniform', 'trench coat', 'sweatshirt', 'cardigan',
        'sunglasses', 'sunglass', 'wig', 'hair slide', 'lipstick', 'mask'
    }
    has_person_hint = any(
        any(hint in label for hint in person_hints)
        for label, _ in objects
    )

    feat_std = features[0].std()
    detail   = "detailed" if feat_std > 0.5 else "clear" if feat_std > 0.25 else "simple"

    # ── build caption ──
    if is_portrait or has_person_hint:
        # It's a person photo
        clothing = [label for label, _ in objects
                    if any(hint in label for hint in person_hints)]
        if clothing:
            caption = f"A {detail} portrait of a person wearing a {clothing[0]}"
            if len(clothing) >= 2:
                caption += f" and a {clothing[1]}"
        else:
            caption = f"A {detail} portrait of a person"

        # add background context from non-clothing objects
        other = [label for label, _ in objects
                 if not any(hint in label for hint in person_hints)]
        if other:
            caption += f", with {other[0]} in the background"
        caption += "."

    elif objects:
        labels = [label for label, _ in objects]
        caption = f"A {detail} image showing a {labels[0]}"
        if len(labels) >= 2:
            caption += f" and a {labels[1]}"
        if len(labels) >= 3:
            caption += f", with {labels[2]} visible"
        caption += "."

    else:
        if g > r and g > b:
            caption = f"An outdoor {detail} scene with green vegetation."
        elif b > r and b > g:
            caption = f"A {detail} scene with blue tones, possibly sky or water."
        elif bright < 0.3:
            caption = "A dark scene with low lighting."
        else:
            caption = f"A {detail} scene captured in the image."

    return caption


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        features = extract_features(pil_img)

        if model_is_trained():
            text   = greedy_caption(features)
            method = "RNN decoder"
        else:
            text   = feature_based_caption(features, pil_img)
            method = "VGG16 + MobileNetV2 scene analysis"

        print(f"[{method}] {text}")

        buf     = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'caption':  text,
            'image':    f'data:image/jpeg;base64,{img_b64}',
            'method':   method
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'vgg16_loaded':         feature_extractor is not None,
        'scene_classifier':     scene_classifier is not None,
        'caption_model_loaded': caption_model is not None,
        'tokenizer_loaded':     tokenizer is not None,
        'model_trained':        model_is_trained()
    })


if __name__ == '__main__':
    print("Image Captioning → http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
