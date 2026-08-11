"""
Create a simple working image captioning model
This will work with your tokenizer
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
import pickle
import numpy as np

# Load tokenizer to get vocab size
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 34

print(f"Vocabulary size: {vocab_size}")
print(f"Max caption length: {max_length}")

# Build a simple image captioning model
# Image feature input (from VGG16)
image_input = Input(shape=(512,), name='image_features')
image_dense = Dense(256, activation='relu')(image_input)
image_dense = Dropout(0.5)(image_dense)

# Caption sequence input
caption_input = Input(shape=(max_length,), name='caption_sequence')
caption_embed = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
caption_lstm = LSTM(256)(caption_embed)
caption_lstm = Dropout(0.5)(caption_lstm)

# Merge image and caption
merged = Add()([image_dense, caption_lstm])
merged = Dense(256, activation='relu')(merged)

# Output layer
output = Dense(vocab_size, activation='softmax')(merged)

# Create model
model = Model(inputs=[image_input, caption_input], outputs=output)

print("\nModel created successfully!")
print(model.summary())

# Save the model
model.save('image_captioning_model.h5', save_format='h5')
print("\n✅ Model saved as 'image_captioning_model.h5'")
print("This is a fresh model (untrained) but it will work with the app!")
print("You can now run: python app.py")
