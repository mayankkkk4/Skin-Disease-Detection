"""
Flask Web Application Server for Skin Disease Detection.
Serves web UI dashboard and REST API endpoints.
"""

import os
import sys
import base64
from io import BytesIO
from PIL import Image

# Suppress TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify, send_from_directory
from disease_db import DISEASES, MODEL_METRICS, CONFUSION_MATRIX
from model_engine import SkinDiseaseModelEngine

app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize Deep Learning / Vision Model Engine
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_engine = SkinDiseaseModelEngine(base_dir=BASE_DIR)

@app.route("/")
def index():
    """Serves the main Skin Disease Detection web application dashboard."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns server status and available models."""
    return jsonify({
        "status": "healthy",
        "service": "Skin Disease AI Diagnostic Server",
        "disease_count": len(DISEASES),
        "keras_models_loaded": list(model_engine.models.keys())
    })

@app.route("/api/diseases", methods=["GET"])
def get_diseases():
    """Returns detailed knowledge base for all 7 skin diseases."""
    return jsonify({
        "success": True,
        "diseases": list(DISEASES.values())
    })

@app.route("/api/models", methods=["GET"])
def get_models():
    """Returns accuracy, precision, and architectural breakdown for CNN models."""
    return jsonify({
        "success": True,
        "metrics": MODEL_METRICS,
        "confusion_matrix": CONFUSION_MATRIX
    })

@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    """Returns list of pre-loaded sample skin lesion images for instant demo testing."""
    samples = []
    samples_dir = os.path.join(BASE_DIR, "static", "samples")
    if os.path.exists(samples_dir):
        for code, info in DISEASES.items():
            filename = f"{code}_sample.jpg"
            if os.path.exists(os.path.join(samples_dir, filename)):
                samples.append({
                    "code": code,
                    "name": info["short_name"],
                    "url": f"/static/samples/{filename}",
                    "risk_level": info["risk_level"],
                    "risk_color": info["risk_color"]
                })
    return jsonify({
        "success": True,
        "samples": samples
    })

@app.route("/api/predict", methods=["POST"])
def predict_lesion():
    """
    Primary API endpoint for skin disease analysis.
    Accepts multipart file upload or JSON with base64 image data.
    """
    try:
        selected_model = request.form.get("model", "Ensemble")
        pil_img = None

        # Option 1: File upload
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            img_bytes = file.read()
            pil_img = Image.open(BytesIO(img_bytes))
            pil_img.load()  # Ensure image data is loaded into memory

        # Option 2: JSON payload with base64 image or sample path
        elif request.is_json:
            data = request.get_json()
            selected_model = data.get("model", selected_model)
            
            if "image_base64" in data and data["image_base64"]:
                b64_str = data["image_base64"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                pil_img = Image.open(BytesIO(img_bytes))
            elif "image_url" in data and data["image_url"]:
                # Load sample image relative path
                rel_path = data["image_url"].lstrip("/")
                full_path = os.path.join(BASE_DIR, rel_path)
                if os.path.exists(full_path):
                    pil_img = Image.open(full_path)

        if pil_img is None:
            return jsonify({
                "success": False,
                "error": "No valid image provided. Please upload an image file or select a sample."
            }), 400

        # Run AI prediction via model engine
        result = model_engine.predict(pil_img, selected_model=selected_model)
        result["success"] = True
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Diagnostic processing failed: {str(e)}"
        }), 500

if __name__ == "__main__":
    print("Starting Skin Disease Detection Web Server on http://127.0.0.1:5000 ...", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
