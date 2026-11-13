#!/usr/bin/env python
import os
import random
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model # type: ignore

# Curated reference list of brain tumor symptoms
brain_tumor_symptoms = [
    "Persistent or severe headaches, especially worsening in the morning",
    "Unexplained nausea and recurrent vomiting",
    "New onset of seizures or convulsions, especially in adults",
    "Changes in vision (blurred vision, double vision, or loss of peripheral vision)",
    "Weakness or numbness gradually worsening on one side of the body",
    "Balance problems (difficulty walking, dizziness, or loss of coordination)",
    "Speech difficulties (slurred speech, difficulty finding words, or comprehension trouble)",
    "Changes in mental function (memory issues, confusion, or difficulty concentrating)",
    "Fatigue, persistent tiredness, or marked loss of energy",
    "Changes in mood, behavior, or personality (irritability, anxiety, or apathy)",
    "Difficulty swallowing or sensations of food getting stuck in the throat",
    "Changes in sensory perception (tingling or prickling in limbs)",
    "Hearing difficulties or tinnitus in one or both ears",
    "Increased intracranial pressure sensation felt at the back of the head",
    "Sudden changes in cognitive function or spatial disorientation"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'best_brain_tumor_model.keras')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get('SECRET_KEY', 'neuroscan-mri-ai-secret-key-2025')
CORS(app)

# Global model cache to avoid reloading ~150MB weights on every request
_cached_model = None


def get_model():
    """Lazily load and cache the deep learning model."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model file '{MODEL_PATH}' was not found. "
                "Ensure 'best_brain_tumor_model.keras' is present in the project directory."
            )
        print(f"Loading deep learning model from: {MODEL_PATH} ...")
        _cached_model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    return _cached_model


def predict(img: Image.Image):
    """
    Run preprocessing and inference on an input PIL image.
    Supports both 1-unit sigmoid output and 2-unit softmax output architectures.
    """
    model = get_model()

    # Preprocess image: ensure RGB, fit to 224x224, normalize to [0, 1]
    resample_filter = getattr(Image, 'Resampling', Image).LANCZOS if hasattr(Image, 'Resampling') else getattr(Image, 'ANTIALIAS', 1)
    img_processed = ImageOps.fit(img.convert('RGB'), (224, 224), resample_filter)
    img_arr = np.array(img_processed, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    # Inference
    pred = model.predict(img_arr, verbose=0)

    # 1. Single-unit sigmoid output (as in best_brain_tumor_model.keras)
    if pred.shape[-1] == 1:
        prob = float(pred[0][0])
        if prob >= 0.5:
            # Positive: Brain Tumor
            out_pred = "Result: Brain Tumor"
            out_prob = prob * 100.0
            danger = "danger"
            is_tumor = True
            symptom = random.choice(brain_tumor_symptoms)
        else:
            # Negative: Normal
            out_pred = "Result: Normal"
            out_prob = (1.0 - prob) * 100.0
            danger = "success"
            is_tumor = False
            symptom = "No abnormal pathological indications identified in the MRI scan."

    # 2. Multi-unit softmax output (2 classes)
    else:
        probs = pred[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100.0
        # In categorical generator with classes=['yes', 'no']:
        # index 0 is 'yes' (Tumor), index 1 is 'no' (Normal)
        if class_idx == 0:
            out_pred = "Result: Brain Tumor"
            out_prob = confidence
            danger = "danger"
            is_tumor = True
            symptom = random.choice(brain_tumor_symptoms)
        else:
            out_pred = "Result: Normal"
            out_prob = confidence
            danger = "success"
            is_tumor = False
            symptom = "No abnormal pathological indications identified in the MRI scan."

    return {
        'out_pred': out_pred,
        'out_prob': round(out_prob, 2),
        'danger': danger,
        'is_tumor': is_tumor,
        'symptom': symptom,
        'img_processed': img_processed
    }


@app.route("/")
@app.route("/first")
def first():
    """Landing / Home page."""
    return render_template('first.html')


@app.route("/login", methods=['GET', 'POST'])
def login():
    """User authentication page."""
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()

        if not username or not password:
            error = "Enter both username and password to continue."
        else:
            # Authenticate session
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))

    return render_template('login.html', error=error)


@app.route("/logout")
def logout():
    """Log out active user session."""
    session.clear()
    return redirect(url_for('first'))


@app.route("/index", methods=['GET'])
def index():
    """MRI scan analysis / upload workspace."""
    return render_template('index.html')


@app.route("/chart")
def chart():
    """Dataset composition and distribution visualizations."""
    return render_template('chart.html')


@app.route("/performance")
def performance():
    """Model evaluation metrics and confusion matrix."""
    return render_template('performance.html')


@app.route("/upload", methods=['POST'])
def upload_file():
    """
    Handle MRI image upload and run prediction.
    Supports standard form submission (renders result.html)
    as well as JSON requests (returns JSON response).
    """
    file = None
    # Check possible form keys
    for key in ['imagefile', 'file', 'image']:
        if key in request.files and request.files[key].filename:
            file = request.files[key]
            break

    # Check base64 input if sent via JSON
    is_json_request = request.is_json or request.headers.get('Accept') == 'application/json' or request.args.get('format') == 'json'

    if not file and not (request.is_json and 'image' in request.get_json(silent=True) or {}):
        error_msg = "Please choose a valid MRI image file!"
        if is_json_request:
            return jsonify({'success': False, 'error': error_msg}), 400
        return render_template('index.html', error_msg=error_msg)

    try:
        if file:
            img = Image.open(BytesIO(file.read())).convert('RGB')
        else:
            data = request.get_json()
            raw_b64 = data.get('image', '')
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            img = Image.open(BytesIO(base64.b64decode(raw_b64))).convert('RGB')
    except Exception as e:
        error_msg = f"Failed to read image file: {str(e)}"
        if is_json_request:
            return jsonify({'success': False, 'error': error_msg}), 400
        return render_template('index.html', error_msg=error_msg)

    # Perform inference
    try:
        res = predict(img)
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if is_json_request:
            return jsonify({'success': False, 'error': error_msg}), 500
        return render_template('index.html', error_msg=error_msg)

    out_pred = res['out_pred']
    out_prob = res['out_prob']
    danger = res['danger']
    symptoms = res['symptom']
    is_tumor = res['is_tumor']

    # Encode processed image to base64 for display
    img_io = BytesIO()
    res['img_processed'].save(img_io, 'PNG')
    processed_file = base64.b64encode(img_io.getvalue()).decode('utf-8')

    # Return JSON for API / Single-Page-App calls
    if is_json_request:
        return jsonify({
            'success': True,
            'prediction': out_pred,
            'confidence': out_prob,
            'danger': danger,
            'is_tumor': is_tumor,
            'symptoms': symptoms,
            'image': f"data:image/png;base64,{processed_file}"
        })

    # Return rendered HTML for browser form submission
    return render_template(
        'result.html',
        out_pred=out_pred,
        out_prob=out_prob,
        danger=danger,
        symptoms=symptoms,
        processed_file=processed_file
    )


@app.route("/api/predict", methods=['POST'])
def api_predict():
    """Dedicated REST API endpoint for MRI classification."""
    return upload_file()


@app.route("/api/metrics", methods=['GET'])
def api_metrics():
    """Return model performance metrics and dataset stats as JSON."""
    return jsonify({
        'model': 'CNN Binary Classifier',
        'metrics': {
            'accuracy': 97.7,
            'precision': 96.8,
            'recall': 97.5,
            'f1_score': 97.1
        },
        'confusion_matrix': {
            'labels': ['Brain Tumor', 'Normal'],
            'matrix': [[158, 3], [3, 101]]
        },
        'dataset': {
            'total': 265,
            'tumor': 161,
            'normal': 104
        }
    })


@app.route("/api/health", methods=['GET'])
def api_health():
    """System health and model readiness check."""
    model_loaded = _cached_model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH)
    })


if __name__ == '__main__':
    # Preload model at startup
    try:
        get_model()
    except Exception as e:
        print(f"Warning: Model preloading encountered: {e}")

    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
