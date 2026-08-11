# Next Word Predictor UI

A simple web application for predicting next words using your LSTM_model.h5 trained on a novel.

## Features

- Clean, modern UI
- Text input for your starting text
- Slider to select number of words to predict (1-15)
- Real-time predictions using your H5 model
- Python Flask backend for model inference

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have these files in the same directory:
   - `LSTM_model.h5` (your trained model)
   - `tokenizer.pkl` (your tokenizer - if you don't have this, you'll need to create it)

## How to Run

1. Start the Python server:
   ```bash
   python server.py
   ```
   The server will run on `http://localhost:5000`

2. Open `index.html` in your web browser (just double-click it)

3. The app will automatically check if the model is loaded

4. Enter some text, adjust the slider (1-15 words), and click "Predict Next Words"

## Creating tokenizer.pkl

If you don't have a `tokenizer.pkl` file, you need to save your tokenizer from training:

```python
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# After training your model, save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
```

## Project Structure

```
├── index.html          # Frontend UI
├── app.js             # Frontend JavaScript
├── style.css          # Styling
├── server.py          # Flask backend server
├── LSTM_model.h5      # Your trained model
├── tokenizer.pkl      # Your tokenizer (needs to be created)
└── requirements.txt   # Python dependencies
```

## Technologies Used

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python, Flask, TensorFlow/Keras
- Model: LSTM (H5 format)
