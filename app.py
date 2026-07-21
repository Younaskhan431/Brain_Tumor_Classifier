import os
import uuid
import logging
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed upload extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

logging.basicConfig(level=logging.INFO)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']  # alphabetical order, matches training
IMG_SIZE = (224, 224)

# All three models were trained on raw [0, 255] pixel input — each one has its own
# rescaling/preprocessing baked in as an internal layer (Rescaling for the CNN,
# mobilenet_v2.preprocess_input / efficientnet.preprocess_input for the other two).
# So the same prepare_image() output can feed all three unchanged.
MODEL_CONFIG = {
    'efficientnet': {
        'path': os.path.join('models', 'efficientnet_model.keras'),
        'label': 'EfficientNetB0',
        'is_best': True,
    },
    'mobilenet': {
        'path': os.path.join('models', 'mobilenet_model.keras'),
        'label': 'MobileNetV2',
        'is_best': False,
    },
    'cnn': {
        'path': os.path.join('models', 'cnn_model.keras'),
        'label': 'Custom CNN',
        'is_best': False,
    },
}

models = {}
model_load_errors = []
for key, cfg in MODEL_CONFIG.items():
    path = cfg['path']
    try:
        # Some models use a Lambda layer referencing `preprocess_input` from
        # their application module. Provide the correct function via
        # `custom_objects` so model deserialization succeeds.
        custom_objects = {}
        if key == 'mobilenet':
            try:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
                custom_objects['preprocess_input'] = mobilenet_preprocess
            except Exception:
                logging.debug('Could not import mobilenet preprocess_input')
        elif key == 'efficientnet':
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
                custom_objects['preprocess_input'] = efficientnet_preprocess
            except Exception:
                logging.debug('Could not import efficientnet preprocess_input')

        if custom_objects:
            models[key] = load_model(path, custom_objects=custom_objects)
        else:
            models[key] = load_model(path)

        logging.info(f"Loaded model {key} from {path}")
    except Exception as e:
        logging.exception(f"Failed to load model {key} from {path}")
        models[key] = None
        model_load_errors.append(f"{key}: {e}")

BEST_MODEL_KEY = next(key for key, cfg in MODEL_CONFIG.items() if cfg['is_best'])

# Expose model load status on app config
app.config['MODELS_OK'] = all(m is not None for m in models.values())
app.config['MODEL_ERRORS'] = model_load_errors


def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # NOTE: no /255 here — each model has its own rescaling/preprocessing layer built in
    return img_array


def predict_with_all_models(img_array):
    """Runs the image through all three models and returns a dict keyed by model key."""
    results = {}
    for key, cfg in MODEL_CONFIG.items():
        model = models.get(key)
        if model is None:
            raise RuntimeError(f"Model '{key}' is not available")
        predictions = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)
        results[key] = {
            'label': cfg['label'],
            'prediction': predicted_class,
            'confidence': confidence,
            'is_best': cfg['is_best'],
        }
    return results


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    # Check models loaded
    if not app.config.get('MODELS_OK', False):
        return render_template('index.html', error='Server error: one or more models failed to load. Check server logs.')

    # Validate file extension
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type. Allowed: PNG, JPG, JPEG')

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    img_array = prepare_image(filepath)
    all_results = predict_with_all_models(img_array)

    primary = all_results[BEST_MODEL_KEY]
    others = [v for k, v in all_results.items() if k != BEST_MODEL_KEY]

    image_url = url_for('static', filename=f"uploads/{unique_name}")

    return render_template(
        'result.html',
        primary=primary,
        others=others,
        image_path=image_url
    )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)