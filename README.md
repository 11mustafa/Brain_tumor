# NeuroScan AI - Brain MRI Deep Learning Classification Platform

**NeuroScan AI** is an end-to-end deep learning medical imaging platform designed for classifying Brain MRI scans into **Brain Tumor** or **Normal** categories with high confidence and sensitivity.

The project features a Convolutional Neural Network (CNN) built with TensorFlow/Keras, an interactive Flask web application, REST API endpoints, and a responsive dark-themed medical HUD interface.

---

## Key Features

1. **AI-Powered MRI Classification**:
   - Deep learning CNN binary classification trained on axial and coronal brain MRI scans.
   - Preprocessing pipeline: RGB conversion, LANCZOS 224x224 normalization, batch tensor transformations.
   - Evaluates classification confidence and references associated clinical symptoms.

2. **Web Application & UI (NeuroScan AI)**:
   - **Home (`/` or `/first`)**: Landing hero with medical imaging intelligence overview.
   - **Login (`/login`)**: Secure session authentication.
   - **Diagnostic Workspace (`/index`)**: Drag-and-drop MRI scan uploader with live preview, laser scanner animation, and 3-step pipeline checklist.
   - **Diagnostic Report (`/upload`)**: Processing viewer HUD with multi-mode view switcher, classification diagnosis (Tumor / Normal), clinical symptoms, and animated SVG confidence gauge.
   - **Model Performance (`/performance`)**: Key performance metrics (Accuracy 97.4%, Precision 99.4%, Recall 96.3%, F-Measure 97.8%) and interactive Confusion Matrix.
   - **Dataset Intelligence (`/chart`)**: Composition analysis and interactive donut distribution chart (161 Tumor scans vs 104 Normal scans).

3. **Explainable AI (Grad-CAM) & Red Tumor Localization**:
   - **Gradient-weighted Class Activation Mapping (Grad-CAM)**: Backpropagates gradients from the un-saturated tumor logit into the MobileNetV2 `out_relu` convolutional feature maps to locate the tumor.
   - **Red Tumor Showcase Overlay**: Prominently highlights the identified tumor area in **vivid red** with a glowing boundary contour for immediate, unambiguous identification.
   - **Multi-View Interactive Switcher**: Instant switching between **Tumor Focus (Red)**, **Heatmap Spectrum (Thermal Jet)**, and **Original MRI Scan**.
   - **Spatial Metrics**: Computes focal region quadrant (e.g. Upper Left Region), estimated lesion coverage area (% of scan), and peak activation intensity.

4. **REST API Endpoints**:
   - `POST /api/predict`: Multipart/form-data or JSON base64 inference endpoint returning prediction, confidence, symptoms, Grad-CAM red overlay, thermal heatmap, and localization coordinates.
   - `GET /api/metrics`: Performance metrics and confusion matrix data in JSON format.
   - `GET /api/health`: Health status and model readiness.

---

## Project Structure

```text
Project/
├── app.py                         # Flask web server & prediction backend
├── best_brain_tumor_model.keras   # Pre-trained CNN deep learning model
├── model.ipynb                    # Model training, augmentation & evaluation notebook
├── requirements.txt               # Python package dependencies
├── test_app.py                    # Automated test suite for routes & predictions
├── Accuracy.txt                   # Recorded model accuracy
├── templates/                     # Flask Jinja2 HTML templates
│   ├── base.html                  # Core layout, navigation & brand HUD
│   ├── first.html                 # Home / Hero landing page
│   ├── login.html                 # Login & authentication page
│   ├── index.html                 # Analysis & MRI upload workspace
│   ├── result.html                # Processing report & confidence gauge
│   ├── performance.html           # Model evaluation & confusion matrix
│   └── chart.html                 # Dataset composition & donut chart
├── static/                        # Static assets
│   ├── css/style.css              # Custom styling & animations
│   └── images/neuroscan-hero.jpg  # Hero background asset
├── tamplete/                      # Frontend workspace & TypeScript components
│   └── src/App.tsx                # React SPA implementation connected to API
└── upload/                        # Sample scans & training dataset
    ├── Brain Tumor/               # Sample positive tumor test scans
    ├── Normal/                    # Sample normal test scans
    └── trained data/              # Training & validation dataset (yes/no)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.9+ (Python 3.10 - 3.13 supported)
- Virtual environment (recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### 1. Launch the Web Server
Run the Flask server:
```bash
python app.py
```

### 2. Open in Browser
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

### 3. Usage Walkthrough
1. From the **Home** page, click **Access workspace** (or log in via `/login` with any username/password, e.g., `admin` / `password`).
2. On the **Analysis** workspace, drag and drop an MRI scan (sample scans are provided in `upload/Brain Tumor` and `upload/Normal`) or click to browse.
3. Click **Execute Diagnostic Pipeline**.
4. Review the generated **AI Analysis Report** with the model's prediction, confidence ring gauge, and reference symptoms.
5. Explore the **Performance** and **Charts** tabs to review model evaluation scores and dataset statistics.

---

## Running Automated Tests

Run the test suite to verify all Flask routes, model inference, and error handling:
```bash
python test_app.py
```

Expected output:
```text
=== Test 1: Upload Brain Tumor Image (Form) ===
Test 1 Passed: Tumor image correctly classified in HTML!
=== Test 2: Upload Normal Image (Form) ===
Test 2 Passed: Normal image correctly classified in HTML!
=== Test 3: API Predict Tumor (JSON) ===
Test 3 Passed! API JSON response: Result: Brain Tumor (89.27%)
=== Test 4: API Predict Normal (JSON) ===
Test 4 Passed! API JSON response: Result: Normal (77.8%)
=== Test 5: Empty file upload ===
Test 5 Passed: Empty file handled gracefully!
ALL PREDICTION AND UPLOAD TESTS PASSED SUCCESSFULLY!
```

---

## Model Training & Jupyter Notebook

The model training notebook `model.ipynb` contains the full pipeline:
- **Dynamic Dataset Discovery**: Automatically detects local dataset paths.
- **Data Augmentation**: Random rotation, translation, zoom, and contrast layers.
- **CNN Architecture**: 3-block convolutional network with pooling and dense layers.
- **Evaluation**: Classification report, confusion matrix, precision, recall, and F1-score.
- **Model Persistence**: Automatically saves the trained weights to `best_brain_tumor_model.keras` and logs accuracy to `Accuracy.txt`.

To run the notebook:
```bash
jupyter notebook model.ipynb
```

