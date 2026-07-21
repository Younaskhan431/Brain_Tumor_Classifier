# Brain Tumor MRI Classifier — Model Comparison App

A local Flask app that runs an uploaded MRI image through all three trained
models (Custom CNN, MobileNetV2, EfficientNetB0) and shows their predictions
side by side.

## 1. Get your trained models onto your machine

Download the three `.keras` files from Google Drive and place them inside
the `models/` folder here, using these exact names (or update the paths at
the top of `app.py` if you'd rather keep your own filenames):

```
models/
  cnn_model.keras
  final_mobilenet_model.keras
  final_efficientnet_model.keras
```

## 2. Set up a virtual environment and install dependencies

```bash
cd brain_tumor_classifier_app
python -m venv venv

# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Check `CLASS_NAMES` in `app.py`

`app.py` has this near the top:

```python
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
```

This **must** match the exact order of `class_names` that was printed in
your training notebooks (Section 3 — the alphabetically sorted folder names
from your dataset). If your dataset's folder names are different, update
this list to match, in the same order.

## 4. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 5. Use it

- Upload an MRI image (`.png`, `.jpg`, `.jpeg`).
- Click "Classify".
- You'll see the predicted class + confidence, plus a full probability
  breakdown, for all three models at once — so you can visually compare
  how they disagree or agree on the same image.

## Notes / things you may want to extend later

- Right now all three models load into memory at startup. If your machine
  is memory-constrained, you can load models lazily per-request instead
  (slower per request, lower idle memory).
- No GPU is required for inference on single images — CPU is fine.
- If you later want to deploy this somewhere public (Render, Railway,
  Hugging Face Spaces, etc.) instead of just running locally, let me know
  and I can walk you through that separately — hosting three TensorFlow
  models publicly has its own considerations (image size limits, cold
  starts, etc.).
