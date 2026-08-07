"""
Preprocessing helpers for each model type.
"""

import numpy as np
import re
import io
from PIL import Image


# ── Image helpers ──────────────────────────────────────────────

def load_pil(file_storage):
    """Read a Werkzeug FileStorage into a PIL Image (RGB)."""
    data = file_storage.read()
    return Image.open(io.BytesIO(data)).convert("RGB"), data


def preprocess_mnist(pil_img):
    """28×28 grayscale → flatten → (1, 784) float32 normalised."""
    img = pil_img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def preprocess_catdog(pil_img):
    """224×224 MobileNet normalisation → (1, 224, 224, 3)."""
    from tensorflow.keras.applications.mobilenet import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    img = pil_img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def preprocess_emotion(pil_img):
    """48×48 grayscale → (1, 48, 48, 1) normalised."""
    import cv2
    import numpy as np
    # Convert PIL → numpy BGR for cv2
    img_np = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (48, 48))
    return gray.reshape(1, 48, 48, 1) / 255.0


def preprocess_caption(pil_img):
    """VGG16 feature extraction → (1, 512)."""
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    import model_loader

    extractor = model_loader._load(
        "vgg16_extractor",
        "__vgg16__",   # sentinel — handled below
        "__vgg16__"
    )
    img = pil_img.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return extractor.predict(arr, verbose=0)


# ── Text helpers ───────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_sentiment(text):
    """IMDB word-index encoding → (1, 200) int32."""
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    VOCAB_SIZE = 10_000
    MAX_LEN = 200

    try:
        word_index = imdb.get_word_index()
    except Exception:
        word_index = {}

    text = clean_text(text)
    seq = []
    for w in text.split():
        idx = word_index.get(w, 0)
        shifted = idx + 3
        seq.append(shifted if shifted < VOCAB_SIZE else 2)
    seq = seq or [2]
    return pad_sequences([seq], maxlen=MAX_LEN, padding="pre").astype(np.int32)


def preprocess_sms(text, tokenizer, max_len):
    """GRU tokenizer → padded sequence."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")


def preprocess_lstm(text, tokenizer, max_len):
    """LSTM tokenizer → padded sequence."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([text])[0]
    return pad_sequences([seq], maxlen=max_len, padding="pre")


def preprocess_pos(sentence, word_index, max_len=50):
    """BiLSTM word-index encoding → (1, 50) padded."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokens = sentence.strip().split()
    seq = [word_index.get(w.lower(), 1) for w in tokens]
    padded = pad_sequences([seq], maxlen=max_len, padding="post", value=0)
    return tokens, padded
