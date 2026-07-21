# Brain Tumor MRI Classifier

A Flask web application for classifying brain MRI scans into four classes using three trained deep learning models and comparing their predictions side by side.

## Overview

This project takes an uploaded MRI scan and predicts one of the following classes:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The app runs three models on each uploaded image so their predictions can be compared directly:

| Model | Approach | Role |
|---|---|---|
| EfficientNetB0 | Transfer learning + fine-tuning | Primary result shown to the user |
| MobileNetV2 | Transfer learning | Comparison model |
| Custom CNN | Built from scratch | Comparison model |

## Features

- Drag-and-drop MRI upload with preview
- Simultaneous inference across all three models
- Highlighted primary prediction with confidence
- Side-by-side comparison of results
- Clean dark-themed interface

## Tech Stack

- Backend: Flask (Python)
- Deep Learning: TensorFlow / Keras
- Frontend: HTML, CSS, and vanilla JavaScript

## Project Structure

```text
Brain_Tumor_Classifier/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── cnn_model.keras
│   ├── mobilenet_model.keras
│   └── efficientnet_model.keras
├── static/
│   ├── style.css
│   └── uploads/
└── templates/
    ├── index.html
    └── result.html
```

## Setup and Installation

1. Clone the repository

```bash
git clone https://github.com/Younaskhan431/Brain_Tumor_Classifier.git
cd Brain_Tumor_Classifier
```

2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Model Details

- Input size: 224 × 224 RGB images
- Class labels expected by the app: `glioma`, `meningioma`, `notumor`, `pituitary`
- Each model includes its own preprocessing inside the model architecture, so the app passes raw image arrays directly to all three models

## Dataset

The models were trained on the Brain Tumor MRI Dataset from Kaggle.

## Author

Younas — Software Engineering student, University of Malakand
Final Year Project, supervised by Dr. Muhammad Faisal

## License

This project is intended for academic and educational purposes.
