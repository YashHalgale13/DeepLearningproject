"""
Lazy model loader — loads each model once on first use.
Supports .h5, .keras, and .pkl formats.
"""

import os
import pickle
import numpy as np

_cache = {}


def _load(key, path, fmt):
    if key in _cache:
        return _cache[key]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if fmt in ("h5", "keras"):
        import tensorflow as tf
        # safe_mode/compile=False helps compatibility across Keras versions.
        try:
            obj = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        except Exception:
            # Older models may contain `quantization_config=None` in layer configs.
            # Newer Keras rejects this kwarg; retry by stripping quantization_config.
            with open(path, "rb") as f:
                raw = f.read()
            # Best-effort fix: remove "quantization_config": null from the HDF5 JSON config.
            # (Some saved models include quantization_config=None which newer Keras rejects.)
            try:
                fixed = raw.replace(b'"quantization_config": null', b'"quantization_config": {"class_name": "None"}')
                # If above doesn't match, just proceed with raw (Keras may still load).
                data_to_write = fixed if fixed != raw else raw
            except Exception:
                data_to_write = raw

            tmp = os.path.join(os.path.dirname(__file__), f"_tmp_{key}_no_quant.h5")
            with open(tmp, "wb") as f:
                f.write(data_to_write)
            obj = tf.keras.models.load_model(tmp, compile=False, safe_mode=False)
    elif fmt == "pkl":
        # Some older PKLs contain a serialized Keras model. Try tf.keras load first.
        try:
            import tensorflow as tf
            obj = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        except Exception:
            # Fallback to raw pickle.
            with open(path, "rb") as f:
                obj = pickle.load(f)

        # MNIST ANN was provided as a pickle. On some environments, unpickling

        # may fail because the object was saved as a Keras model with fields
        # not recognized by the current Keras version.
        # If pickle fails, try loading via tf.keras.
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except Exception:
                # Older Keras objects may be stored inside the pickle. If both
                # pickle and tf.keras deserialization fail, surface the error.
                raise






    else:
        raise ValueError(f"Unknown format: {fmt}")

    _cache[key] = obj
    print(f"[loader] {key} loaded from {os.path.basename(path)}")
    return obj


def get_model(key, path, fmt):
    cache_key = f"model:{key}"

    # If a model was already loaded, return the cached object.
    if cache_key in _cache:
        return _cache[cache_key]

    # In this dashboard we may pass either a path string or a file-like object
    # (from some legacy code paths). Normalize to a filesystem path where possible.
    if hasattr(path, "read") and not isinstance(path, (str, bytes, os.PathLike)):
        # We cannot reliably get a path from a buffered stream.
        # Save it to a temporary file for keras loading.
        data = path.read()
        tmp_path = os.path.join(os.path.dirname(__file__), f"_tmp_{key}.model")
        with open(tmp_path, "wb") as f:
            f.write(data)
        path = tmp_path

    # Keras compatibility: some serialized models include
    # quantization_config=None which newer Keras rejects.
    # Best-effort workaround: load with compile=False and ignore failures.
    try:
        obj = _load(cache_key, path, fmt)
        return obj
    except Exception:
        if fmt in ("h5", "keras"):
        # Retry by loading in a more forgiving way.
            # NOTE: Some of your saved models were created with an older Keras that
            # included `quantization_config=None` inside layer configs.
            # Newer Keras may fail deserialization with:
            #   Unrecognized keyword arguments passed to Dense/Embedding...
            # Best-effort: retry with safe_mode and custom object scope.
            import tensorflow as tf
            obj = tf.keras.models.load_model(path, compile=False, safe_mode=False)
            _cache[cache_key] = obj
            return obj

        if fmt == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)

        raise




def get_tokenizer(key, path):
    cache_key = f"tok:{key}"
    if cache_key in _cache:
        return _cache[cache_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    with open(path, "rb") as f:
        tok = pickle.load(f)
    _cache[cache_key] = tok
    print(f"[loader] tokenizer:{key} loaded")
    return tok
