"""
Unified ML Dashboard — Flask backend
Serves all 8 models under /predict/<model_id>
"""

import os, sys, base64, io, pickle, re
import numpy as np
from flask import Flask, render_template, request, jsonify

# Make sure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Make EX 11 & 12 backend importable for BERT model + attention utils
_EX11_BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "EX 11 & 12", "backend")
if _EX11_BACKEND not in sys.path:
    sys.path.insert(0, _EX11_BACKEND)

import config
import model_loader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024   # 32 MB

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def pil_to_b64(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────
# Main route
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models_meta = {k: {"name": v["name"], "type": v["type"], "description": v["description"]}
                   for k, v in config.MODELS.items()}
    return render_template("index.html", models=models_meta)


# ─────────────────────────────────────────────────────────────
# /predict/mnist
# ─────────────────────────────────────────────────────────────

@app.route("/predict/mnist", methods=["POST"])
def predict_mnist():
    try:
        cfg = config.MODELS["mnist"]
        mdl = model_loader.get_model("mnist", cfg["model_path"], cfg["model_format"])

        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        from PIL import Image
        data = file.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")

        # Preprocess: 28×28 grayscale — try (1,28,28) first, fall back to (1,784)
        img = pil.convert("L").resize((28, 28))
        arr = np.array(img, dtype=np.float32) / 255.0

        # Detect expected input shape from model
        input_shape = mdl.input_shape  # e.g. (None,28,28) or (None,784)
        if len(input_shape) == 3:          # (None, 28, 28)
            arr = arr.reshape(1, 28, 28)
        elif len(input_shape) == 4:        # (None, 28, 28, 1) — CNN
            arr = arr.reshape(1, 28, 28, 1)
        else:                              # (None, 784) — flatten
            arr = arr.reshape(1, -1)

        # sklearn pkl model
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(arr)[0]
            pred = int(np.argmax(proba))
            conf = round(float(np.max(proba)) * 100, 2)
        elif hasattr(mdl, "predict"):
            raw = mdl.predict(arr)
            if hasattr(raw, "__len__") and len(np.array(raw).shape) > 1:
                pred = int(np.argmax(raw[0]))
                conf = round(float(np.max(raw[0])) * 100, 2)
            else:
                pred = int(raw[0])
                conf = 100.0
        else:
            return jsonify({"error": "Unsupported model type"}), 500

        return jsonify({
            "prediction": str(pred),
            "confidence": conf,
            "image": pil_to_b64(pil)
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/catdog
# ─────────────────────────────────────────────────────────────

@app.route("/predict/catdog", methods=["POST"])
def predict_catdog():
    try:
        cfg = config.MODELS["catdog"]
        mdl = model_loader.get_model("catdog", cfg["model_path"], cfg["model_format"])

        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        from PIL import Image
        from tensorflow.keras.applications.mobilenet import preprocess_input
        from tensorflow.keras.preprocessing.image import img_to_array

        data = file.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")

        img = pil.resize((224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        preds = mdl.predict(arr, verbose=0)
        preds = np.array(preds)

        if preds.shape[-1] == 1:
            score = float(preds[0][0])
            label = "Dog" if score > 0.5 else "Cat"
            conf = round((score if label == "Dog" else 1 - score) * 100, 2)
        else:
            idx = int(np.argmax(preds[0]))
            label = ["Cat", "Dog"][idx]
            conf = round(float(preds[0][idx]) * 100, 2)

        return jsonify({"prediction": label, "confidence": conf, "image": pil_to_b64(pil)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/emotion
# ─────────────────────────────────────────────────────────────

@app.route("/predict/emotion", methods=["POST"])
def predict_emotion():
    try:
        cfg = config.MODELS["emotion"]
        mdl = model_loader.get_model("emotion", cfg["model_path"], cfg["model_format"])

        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        import cv2
        from PIL import Image

        EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        data = file.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")

        img_np = np.array(pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (48, 48))
        arr = gray.reshape(1, 48, 48, 1) / 255.0

        preds = mdl.predict(arr, verbose=0)
        idx = int(np.argmax(preds[0]))
        label = EMOTIONS[idx] if idx < len(EMOTIONS) else str(idx)
        conf = round(float(np.max(preds[0])) * 100, 2)

        return jsonify({"prediction": label, "confidence": conf, "image": pil_to_b64(pil)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/sentiment
# ─────────────────────────────────────────────────────────────

@app.route("/predict/sentiment", methods=["POST"])
def predict_sentiment():
    try:
        cfg = config.MODELS["sentiment"]
        mdl = model_loader.get_model("sentiment", cfg["model_path"], cfg["model_format"])

        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        from tensorflow.keras.datasets import imdb
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        VOCAB_SIZE = 10_000
        MAX_LEN = int(mdl.input_shape[1] or 500)
        try:
            word_index = imdb.get_word_index()
        except Exception:
            word_index = {}

        cleaned = clean(text)
        seq = []
        for w in cleaned.split():
            idx = word_index.get(w, 0)
            shifted = idx + 3
            seq.append(shifted if shifted < VOCAB_SIZE else 2)
        seq = seq or [2]
        padded = pad_sequences([seq], maxlen=MAX_LEN, padding="pre").astype(np.int32)

        score = float(mdl.predict(padded, verbose=0).flatten()[0])
        score = max(0.0, min(1.0, score))
        sentiment = "POSITIVE" if score >= 0.5 else "NEGATIVE"
        conf = round((score if sentiment == "POSITIVE" else 1 - score) * 100, 2)

        return jsonify({"sentiment": sentiment, "confidence": conf, "raw_score": round(score, 4)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/caption
# ─────────────────────────────────────────────────────────────

@app.route("/predict/caption", methods=["POST"])
def predict_caption():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        from PIL import Image
        from tensorflow.keras.applications import VGG16, MobileNetV2
        from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mn_pre
        from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
        from tensorflow.keras.preprocessing.image import img_to_array

        data = file.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")

        # VGG16 extractor (cached)
        if "vgg16" not in model_loader._cache:
            model_loader._cache["vgg16"] = VGG16(weights="imagenet", include_top=False, pooling="avg")
        extractor = model_loader._cache["vgg16"]

        # MobileNetV2 scene classifier (cached)
        if "mobilenetv2" not in model_loader._cache:
            model_loader._cache["mobilenetv2"] = MobileNetV2(weights="imagenet", include_top=True)
        scene_clf = model_loader._cache["mobilenetv2"]

        # Extract VGG16 features
        img224 = pil.resize((224, 224))
        arr = img_to_array(img224)
        arr = np.expand_dims(arr, axis=0)
        features = extractor.predict(vgg_pre(arr.copy()), verbose=0)

        # MobileNetV2 top-5 predictions
        mn_arr = mn_pre(arr.copy())
        preds_mn = scene_clf.predict(mn_arr, verbose=0)
        top5 = decode_predictions(preds_mn, top=5)[0]
        objects = [(lbl.replace("_", " "), float(sc)) for (_, lbl, sc) in top5 if sc > 0.03]

        # Color / brightness analysis
        small = np.array(pil.resize((64, 64)).convert("RGB")) / 255.0
        pixels = small.reshape(-1, 3)
        r, g, b = pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean()
        bright = pixels.mean()
        feat_std = features[0].std()
        detail = "detailed" if feat_std > 0.5 else "clear" if feat_std > 0.25 else "simple"

        # Portrait detection
        w, h = pil.size
        skin_mask = (pixels[:, 0] > 0.4) & (pixels[:, 0] > pixels[:, 1]) & (pixels[:, 1] > pixels[:, 2])
        is_portrait = (w / h < 1.2) and (skin_mask.mean() > 0.15)
        person_hints = {"suit","tie","bow tie","lab coat","jersey","shirt","jacket","dress",
                        "gown","uniform","sweatshirt","cardigan","sunglasses","wig","lipstick","mask"}
        has_person = any(any(h in lbl for h in person_hints) for lbl, _ in objects)

        if is_portrait or has_person:
            clothing = [lbl for lbl, _ in objects if any(h in lbl for h in person_hints)]
            caption = f"A {detail} portrait of a person"
            if clothing:
                caption += f" wearing a {clothing[0]}"
            other = [lbl for lbl, _ in objects if not any(h in lbl for h in person_hints)]
            if other:
                caption += f", with {other[0]} in the background"
            caption += "."
        elif objects:
            labels = [lbl for lbl, _ in objects]
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

        return jsonify({"caption": caption, "method": "VGG16 + MobileNetV2", "image": pil_to_b64(pil)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/pos
# ─────────────────────────────────────────────────────────────

@app.route("/predict/pos", methods=["POST"])
def predict_pos():
    try:
        cfg = config.MODELS["pos"]
        mdl = model_loader.get_model("pos", cfg["model_path"], cfg["model_format"])

        word_idx_path = os.path.join(os.path.dirname(cfg["model_path"]), "word_index.pkl")
        tag_idx_path  = os.path.join(os.path.dirname(cfg["model_path"]), "tag_index.pkl")

        word_index = model_loader.get_tokenizer("pos_word", word_idx_path)
        tag_index  = model_loader.get_tokenizer("pos_tag",  tag_idx_path)
        idx_to_tag = {v: k for k, v in tag_index.items()}

        TAG_DESC = {
            "CC":"Coordinating conjunction","CD":"Cardinal number","DT":"Determiner",
            "EX":"Existential there","FW":"Foreign word","IN":"Preposition/subord. conj.",
            "JJ":"Adjective","JJR":"Adjective, comparative","JJS":"Adjective, superlative",
            "MD":"Modal verb","NN":"Noun, singular","NNS":"Noun, plural",
            "NNP":"Proper noun, singular","NNPS":"Proper noun, plural",
            "PRP":"Personal pronoun","RB":"Adverb","RP":"Particle","TO":"to",
            "UH":"Interjection","VB":"Verb, base form","VBD":"Verb, past tense",
            "VBG":"Verb, gerund","VBN":"Verb, past participle",
            "VBP":"Verb, non-3rd person singular","VBZ":"Verb, 3rd person singular",
            "WDT":"Wh-determiner","WP":"Wh-pronoun","WRB":"Wh-adverb",
            ".":"Punctuation",",":"Comma",":":"Colon",
        }

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        data = request.get_json()
        sentence = (data or {}).get("text", "").strip()
        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        MAX_LEN = 50
        tokens = sentence.split()
        seq = [word_index.get(w.lower(), 1) for w in tokens]
        padded = pad_sequences([seq], maxlen=MAX_LEN, padding="post", value=0)
        preds = mdl.predict(padded, verbose=0)[0]

        result = []
        for i, token in enumerate(tokens):
            tag_idx = int(np.argmax(preds[i]))
            tag = idx_to_tag.get(tag_idx, "NN")
            if tag == "<PAD>":
                tag = "NN"
            result.append({
                "word": token,
                "tag": tag,
                "description": TAG_DESC.get(tag, tag),
                "confidence": round(float(np.max(preds[i])) * 100, 1)
            })

        return jsonify({"tokens": result, "count": len(result)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/lstm
# ─────────────────────────────────────────────────────────────

@app.route("/predict/lstm", methods=["POST"])
def predict_lstm():
    try:
        cfg = config.MODELS["lstm"]
        mdl = model_loader.get_model("lstm", cfg["model_path"], cfg["model_format"])
        tok = model_loader.get_tokenizer("lstm", cfg["tokenizer_path"])

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        num_words = int((data or {}).get("num_words", 5))
        if not text:
            return jsonify({"error": "No text provided"}), 400

        max_len = mdl.input_shape[1]
        current = text
        predictions = []

        for _ in range(num_words):
            seq = tok.texts_to_sequences([current])[0]
            seq = pad_sequences([seq], maxlen=max_len, padding="pre")
            pred = mdl.predict(seq, verbose=0)
            widx = int(np.argmax(pred, axis=-1)[0])
            word = next((w for w, i in tok.word_index.items() if i == widx), "[unknown]")
            predictions.append(word)
            current += " " + word

        return jsonify({
            "predictions": predictions,
            "original_text": text,
            "completed_text": current
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/sms
# ─────────────────────────────────────────────────────────────

@app.route("/predict/sms", methods=["POST"])
def predict_sms():
    try:
        cfg = config.MODELS["sms"]
        mdl = model_loader.get_model("sms", cfg["model_path"], cfg["model_format"])
        tok = model_loader.get_tokenizer("sms", cfg["tokenizer_path"])

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        data = request.get_json()
        text = (data or {}).get("text", "").strip()
        if not text:
            return jsonify({"error": "No SMS text provided"}), 400

        max_len = mdl.input_shape[1]
        cleaned = clean(text)
        seq = tok.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="pre", truncating="pre")

        score = float(mdl.predict(padded, verbose=0).flatten()[0])
        label = "SPAM" if score >= 0.5 else "HAM"
        conf = round((score if label == "SPAM" else 1 - score) * 100, 2)

        return jsonify({"label": label, "confidence": conf, "raw_score": round(score, 4)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /predict/fakenews
# ─────────────────────────────────────────────────────────────

@app.route("/predict/fakenews", methods=["POST"])
def predict_fakenews():
    try:
        import torch
        from transformers import BertTokenizer
        from model import FakeNewsClassifier, load_model
        from attention_utils import (
            extract_attention,
            get_meaningful_tokens,
            normalize_attention,
            attention_to_json,
        )

        cfg    = config.MODELS["fakenews"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Lazy-load model ──────────────────────────────────────────────────
        model_cache_key = "model:fakenews"
        if model_cache_key not in model_loader._cache:
            if not os.path.exists(cfg["model_path"]):
                return jsonify({
                    "error": (
                        "BERT model not found. "
                        "Train first and place best_model.pt in EX 11 & 12/models/."
                    )
                }), 503
            model_loader._cache[model_cache_key] = load_model(cfg["model_path"], device)
        bert_model = model_loader._cache[model_cache_key]

        # ── Lazy-load tokenizer ──────────────────────────────────────────────
        tok_cache_key = "tok:fakenews"
        if tok_cache_key not in model_loader._cache:
            model_loader._cache[tok_cache_key] = BertTokenizer.from_pretrained(
                cfg["tokenizer_name"]
            )
        tokenizer = model_loader._cache[tok_cache_key]

        # ── Parse request ────────────────────────────────────────────────────
        data  = request.get_json(force=True) or {}
        text  = data.get("text", "").strip()
        layer = int(data.get("layer", cfg["default_layer"]))
        head  = int(data.get("head",  cfg["default_head"]))

        if len(text) < 10:
            return jsonify({"error": "Text too short. Please enter at least 10 characters."}), 400

        # ── Tokenize ─────────────────────────────────────────────────────────
        encoding = tokenizer(
            text,
            max_length=cfg["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_type_ids = encoding["token_type_ids"].to(device)

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            logits, attentions = bert_model(input_ids, attention_mask, token_type_ids)

        probs      = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx   = int(probs.argmax())
        confidence = round(float(probs[pred_idx]) * 100, 2)
        prediction = {0: "Real", 1: "Fake"}[pred_idx]

        # ── Attention extraction ─────────────────────────────────────────────
        tokens         = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
        raw_matrix     = extract_attention(attentions, layer=layer, head=head)
        trimmed_tokens, trimmed_matrix = get_meaningful_tokens(tokens, raw_matrix)
        norm_matrix    = normalize_attention(trimmed_matrix)
        attn_data      = attention_to_json(trimmed_tokens, norm_matrix)

        return jsonify({
            "prediction":       prediction,
            "confidence":       confidence,
            "tokens":           attn_data["tokens"],
            "attention_matrix": attn_data["matrix"],
            "selected_layer":   layer,
            "selected_head":    head,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "running", "models": list(config.MODELS.keys())})


@app.route("/diagnostics")
def diagnostics():
    """Validate model files + critical dependencies so the UI can show friendly errors."""
    checks = []
    errors = []

    def _safe_add_error(component, hint, err):
        item = {"component": component, "hint": hint}
        if err is not None:
            item["error"] = str(err)
        errors.append(item)


    # Framework availability checks
    for lib, hint in [
        ("tensorflow", "TensorFlow is required for .h5/.keras models"),
        ("PIL", "Pillow is required for image routes"),
        ("cv2", "OpenCV is required for emotion route"),
    ]:

        try:
            if lib == "PIL":
                import PIL  # noqa: F401
            elif lib == "cv2":
                import cv2  # noqa: F401
            else:
                __import__(lib)
            checks.append({"component": lib, "ok": True})
        except Exception as e:
            checks.append({"component": lib, "ok": False, "error": str(e)})
            # Not all libs are strictly required for all routes; still record as potential issue.
            errors.append({"component": lib, "hint": hint, "error": str(e)})

    # Model file checks (from config)
    for mid, cfg in config.MODELS.items():
        model_path = cfg.get("model_path")
        model_format = cfg.get("model_format")
        try:
            ok = bool(model_path) and os.path.exists(model_path)
            checks.append({
                "component": f"model:{mid}",
                "ok": ok,
                "model_path": model_path,
                "model_format": model_format,
            })
            if not ok:
                errors.append({"component": f"model:{mid}", "hint": "Place the trained model file at the configured path."})

            # Tokenizer checks for relevant models
            tok_path = cfg.get("tokenizer_path")
            if tok_path:
                tok_ok = os.path.exists(tok_path)
                checks.append({"component": f"tokenizer:{mid}", "ok": tok_ok, "tokenizer_path": tok_path})
                if not tok_ok:
                    errors.append({"component": f"tokenizer:{mid}", "hint": "Place the tokenizer/indices file at the configured path."})

            # Some models rely on extra index files next to model_path (POS)
            if mid == "pos":
                base_dir = os.path.dirname(cfg["model_path"])
                for fname in ["word_index.pkl", "tag_index.pkl"]:
                    p = os.path.join(base_dir, fname)
                    ok2 = os.path.exists(p)
                    checks.append({"component": "pos_index:" + fname, "ok": ok2, "path": p})
                    if not ok2:
                        errors.append({"component": "pos_index:" + fname, "hint": "Missing POS index file."})

        except Exception as e:
            checks.append({"component": f"model:{mid}", "ok": False, "error": str(e)})
            errors.append({"component": f"model:{mid}", "error": str(e)})

    overall_ok = len([c for c in checks if not c.get("ok") and not c.get("component", "").startswith("model:")]) == 0

    # The dashboard UI currently supports 8 models and does not require
    # fake-news (torch/transformers). So do NOT fail diagnostics for
    # missing torch/transformers.
    required_components_fail = {
        "tensorflow", "PIL", "cv2"
    }

    # If any required core lib is missing => 503
    for e in errors:
        comp = e.get("component", "")
        if comp in required_components_fail:
            return jsonify({
                "status": "needs_attention",
                "checks": checks,
                "errors": errors,
                "models": list(config.MODELS.keys())
            }), 503

    # Otherwise, still return details but UI can proceed.
    return jsonify({
        "status": "ok",
        "checks": checks,
        "errors": errors,
        "models": list(config.MODELS.keys())
    }), 200




if __name__ == "__main__":
    print("ML Dashboard → http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
