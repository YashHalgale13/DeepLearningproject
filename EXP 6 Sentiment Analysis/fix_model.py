"""
Script to fix and re-save the image captioning model
Run this once to convert your model to a compatible format
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer

# Define all custom layers
class TrueDivide(Layer):
    def __init__(self, **kwargs):
        super(TrueDivide, self).__init__(**kwargs)
    
    def call(self, inputs):
        if isinstance(inputs, list):
            return tf.math.truediv(inputs[0], inputs[1])
        return inputs
    
    def get_config(self):
        return super(TrueDivide, self).get_config()

class NotEqual(Layer):
    def __init__(self, **kwargs):
        super(NotEqual, self).__init__(**kwargs)
    
    def call(self, inputs):
        if isinstance(inputs, list):
            return tf.math.not_equal(inputs[0], inputs[1])
        return inputs
    
    def get_config(self):
        return super(NotEqual, self).get_config()

class ExpandDims(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDims, self).__init__(**kwargs)
        self.axis = axis
    
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)
    
    def get_config(self):
        config = super(ExpandDims, self).get_config()
        config.update({'axis': self.axis})
        return config

class OnesLike(Layer):
    def __init__(self, **kwargs):
        super(OnesLike, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.ones_like(inputs)
    
    def get_config(self):
        return super(OnesLike, self).get_config()

class BroadcastTo(Layer):
    def __init__(self, shape=None, **kwargs):
        super(BroadcastTo, self).__init__(**kwargs)
        self.target_shape = shape
    
    def call(self, inputs):
        if self.target_shape:
            return tf.broadcast_to(inputs, self.target_shape)
        return inputs
    
    def get_config(self):
        config = super(BroadcastTo, self).get_config()
        config.update({'shape': self.target_shape})
        return config

class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super(Cast, self).__init__(**kwargs)
        self.target_dtype = dtype
    
    def call(self, inputs):
        return tf.cast(inputs, self.target_dtype)
    
    def get_config(self):
        config = super(Cast, self).get_config()
        config.update({'dtype': self.target_dtype})
        return config

print("Attempting to load and re-save model...")

model_path = 'image_captioning_model.h5'
new_model_path = 'image_captioning_model_fixed.h5'

custom_objects = {
    'TrueDivide': TrueDivide,
    'NotEqual': NotEqual,
    'ExpandDims': ExpandDims,
    'OnesLike': OnesLike,
    'BroadcastTo': BroadcastTo,
    'Cast': Cast,
}

# Add TensorFlow operations
tf_ops = {
    'tf': tf,
    'truediv': tf.math.truediv,
    'not_equal': tf.math.not_equal,
    'expand_dims': tf.expand_dims,
    'ones_like': tf.ones_like,
    'cast': tf.cast,
    'multiply': tf.multiply,
    'subtract': tf.subtract,
    'add': tf.add,
}
custom_objects.update(tf_ops)

try:
    # Try loading with custom objects
    print("Loading model with custom objects...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    print("✅ Model loaded successfully!")
    
    # Re-save in a cleaner format
    print(f"Re-saving model as: {new_model_path}")
    model.save(new_model_path, save_format='h5')
    print(f"✅ Model re-saved as: {new_model_path}")
    print("\n✅ SUCCESS! Now update app.py to use 'image_captioning_model_fixed.h5'")
    print("Or rename the file:")
    print("  del image_captioning_model.h5")
    print("  ren image_captioning_model_fixed.h5 image_captioning_model.h5")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nThe model contains TensorFlow operations that can't be serialized.")
    print("You need to either:")
    print("1. Provide the original model training/building code")
    print("2. Re-train the model and save it properly")
    print("3. Convert to TensorFlow SavedModel format instead of H5")
