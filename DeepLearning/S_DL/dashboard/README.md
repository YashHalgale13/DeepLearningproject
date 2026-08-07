# ML DASHBOARD — BRUTALIST EDITION

Unified Flask dashboard for all 8 ML models.

## Structure

```
dashboard/
├── app.py            ← Flask app, all /predict/<model> routes
├── config.py         ← Model registry (paths, types, metadata)
├── model_loader.py   ← Lazy loader for .h5 / .keras / .pkl
├── preprocess.py     ← Preprocessing helpers (unused directly, kept for reference)
├── requirements.txt
├── templates/
│   └── index.html    ← Brutalist UI
└── static/
    ├── style.css
    └── app.js
```

## Run

```bash
cd dashboard
pip install -r requirements.txt
python app.py
```

Then open: http://localhost:5000

## Models

| # | ID        | Type  | Model File                        |
|---|-----------|-------|-----------------------------------|
| 1 | mnist     | image | ANNModel (1).pkl                  |
| 2 | catdog    | image | cat_dog_MobileNet.h5              |
| 3 | emotion   | image | emotion_model.h5                  |
| 4 | sentiment | text  | imdb_rnn_model (1).h5             |
| 5 | caption   | image | VGG16 + MobileNetV2 (ImageNet)    |
| 6 | pos       | text  | pos_model.h5 + word/tag index     |
| 7 | lstm      | text  | LSTM_model.h5 + tokenizer.pickle  |
| 8 | sms       | text  | gru_model.keras + GRU_Tokenizer   |
