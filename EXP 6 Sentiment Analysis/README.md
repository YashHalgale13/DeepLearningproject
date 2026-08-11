# 🖼️ Image Captioning with Deep Learning

A Flask-based web application that generates descriptive captions for images using a deep learning model with encoder-decoder architecture (one image → many words).

## 🏗️ Architecture

This project uses a **one-to-many** sequence model:
- **Encoder**: VGG16 extracts visual features from the image (one input)
- **Decoder**: LSTM generates caption word by word (many outputs)

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- Pillow (PIL)
- NumPy

## 🚀 Setup Instructions

### Step 1: Add Your Model Files

You need to add two files to the project directory:

1. **image_captioning_model.h5** - Your trained caption generation model
2. **tokenizer.pkl** - The tokenizer used during training

```
project/
├── app.py
├── image_captioning_model.h5  ← Add this
├── tokenizer.pkl               ← Add this
├── requirements.txt
└── templates/
    └── index.html
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Run the Application

```bash
python app.py
```

You should see:
```
🖼️  Image Captioning Server Starting...
📍 Open your browser at: http://localhost:5000
```

### Step 4: Open in Browser

Go to: **http://localhost:5000**

## 📝 How to Use

1. **Upload an image** by clicking the upload area or dragging and dropping
2. **Preview** your image
3. **Click "Generate Caption"** button
4. **View the AI-generated caption** describing your image

## 🎯 Model Requirements

Your model should follow this architecture:

### Input
- **Image features**: Shape (batch_size, 512) from VGG16
- **Partial caption**: Shape (batch_size, max_length) - tokenized sequence

### Output
- **Next word prediction**: Shape (batch_size, vocab_size)

### Training Format
```python
# Encoder: VGG16 features
image_features = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Decoder: LSTM for caption generation
# Input 1: Image features
# Input 2: Partial caption sequence
# Output: Next word probabilities
```

## 📊 Example Captions

**Input Image**: A dog playing in a park
**Generated Caption**: "a brown dog is running through the grass"

**Input Image**: A sunset over mountains
**Generated Caption**: "the sun is setting over the mountains"

## 🔧 Configuration

Edit `app.py` to adjust these settings:

```python
max_caption_length = 34  # Maximum words in caption
vocab_size = 5000        # Size of your vocabulary
```

## 📁 File Structure

```
.
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── image_captioning_model.h5   # Your trained model (add this)
├── tokenizer.pkl               # Your tokenizer (add this)
└── templates/
    └── index.html              # Web interface
```

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Home page with interface |
| `/caption` | POST | Upload image & get caption |
| `/health` | GET | Check server status |

### Example POST Request to `/caption`:

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/caption
```

### Response:
```json
{
  "caption": "a dog is running through the grass",
  "image": "data:image/jpeg;base64,...",
  "filename": "photo.jpg"
}
```

## ⚠️ Troubleshooting

### Model Not Found Error
- Make sure `image_captioning_model.h5` is in the same folder as `app.py`
- Ensure `tokenizer.pkl` exists with your trained tokenizer

### Poor Captions
- Check if your model was trained properly
- Verify the tokenizer matches your training data
- Ensure max_caption_length matches your training configuration

### Image Upload Fails
- Check file size (max 16MB)
- Ensure image format is supported (JPG, PNG, GIF, BMP)

## 🎓 Training Your Own Model

If you need to train a model, here's the basic structure:

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.applications import VGG16

# Image encoder
image_input = Input(shape=(512,))
image_dense = Dense(256, activation='relu')(image_input)

# Caption decoder
caption_input = Input(shape=(max_length,))
caption_embed = Embedding(vocab_size, 256)(caption_input)
caption_lstm = LSTM(256)(caption_embed)

# Merge and predict
merged = Add()([image_dense, caption_lstm])
output = Dense(vocab_size, activation='softmax')(merged)

model = Model(inputs=[image_input, caption_input], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

## 📞 Support

Common issues:
1. Install all dependencies: `pip install -r requirements.txt`
2. Verify model and tokenizer files exist
3. Check model input/output shapes match the code
4. Test with sample images first

---

**Architecture**: Encoder-Decoder (One-to-Many)
**Encoder**: VGG16 (Image Features)
**Decoder**: LSTM (Caption Generation)
