"""
Blood Group Prediction — Flask Web App
Run:  python app.py
Then: open http://localhost:5000
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ── Load Model & Class Map ────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("MODEL_PATH", "models/bloodgroup_cnn.h5")
CLASS_PATH  = os.environ.get("CLASS_PATH",  "models/class_indices.json")
IMG_SIZE    = (128, 128)

model       = None
idx_to_class = {}

def load_model():
    global model, idx_to_class
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅  Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️   Model file not found at {MODEL_PATH}. Run train.py first.")

    if os.path.exists(CLASS_PATH):
        with open(CLASS_PATH) as f:
            idx_to_class = json.load(f)
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    else:
        # Fallback default order
        idx_to_class = {0:"A+",1:"A-",2:"AB+",3:"AB-",4:"B+",5:"B-",6:"O+",7:"O-"}


def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, *IMG_SIZE, 1)


# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Blood Group Predictor</title>
  <style>
    :root {
      --bg:      #0d1117;
      --surface: #161b22;
      --border:  #30363d;
      --red:     #e63946;
      --red-dim: #7c1f27;
      --text:    #e6edf3;
      --muted:   #8b949e;
      --radius:  14px;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem;
    }

    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .drop-icon { font-size: 3rem; margin-bottom: .5rem; }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.5px;
    }
    h1 span { color: var(--red); }
    p.sub { color: var(--muted); margin-top: .4rem; font-size: .95rem; }

    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      width: 100%;
      max-width: 520px;
      padding: 2rem;
    }

    .upload-zone {
      border: 2px dashed var(--border);
      border-radius: 10px;
      padding: 2.5rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
      position: relative;
    }
    .upload-zone:hover, .upload-zone.drag { border-color: var(--red); background: rgba(230,57,70,.05); }
    .upload-zone input[type=file] {
      position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
    }
    .upload-zone .icon { font-size: 2.5rem; }
    .upload-zone p { color: var(--muted); margin-top: .5rem; font-size: .9rem; }
    .upload-zone strong { color: var(--red); }

    #preview-wrap { margin-top: 1.2rem; display: none; text-align: center; }
    #preview-wrap img {
      max-height: 200px;
      border-radius: 8px;
      border: 1px solid var(--border);
      filter: grayscale(1);
    }

    .btn {
      display: block;
      width: 100%;
      margin-top: 1.4rem;
      padding: .85rem;
      background: var(--red);
      color: #fff;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: opacity .2s, transform .1s;
    }
    .btn:hover { opacity: .88; }
    .btn:active { transform: scale(.98); }
    .btn:disabled { opacity: .45; cursor: not-allowed; }

    #result {
      margin-top: 1.6rem;
      display: none;
    }
    .result-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: 1px; }
    .blood-type {
      font-size: 3.5rem;
      font-weight: 800;
      color: var(--red);
      line-height: 1;
      margin: .3rem 0;
    }
    .confidence { color: var(--muted); font-size: .95rem; }
    .bar-wrap { margin-top: 1.2rem; }
    .bar-row { display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem; }
    .bar-label { width: 3rem; font-size: .8rem; color: var(--muted); text-align: right; }
    .bar-bg { flex: 1; background: var(--border); border-radius: 4px; height: 8px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--red); border-radius: 4px; transition: width .6s ease; }
    .bar-pct { width: 3.2rem; font-size: .78rem; color: var(--muted); }

    .error-msg { color: #f97316; margin-top: 1rem; font-size: .9rem; }
    .spinner { display:none; width:20px;height:20px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;margin:auto; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<header>
  <div class="drop-icon">🩸</div>
  <h1>Blood Group <span>Predictor</span></h1>
  <p class="sub">Upload a fingerprint image — CNN will predict the blood group</p>
</header>

<div class="card">
  <div class="upload-zone" id="zone">
    <input type="file" id="fileInput" accept="image/*"/>
    <div class="icon">🖐</div>
    <p>Drag & drop or <strong>browse</strong> a fingerprint image</p>
    <p style="margin-top:.3rem">PNG · BMP · JPG</p>
  </div>

  <div id="preview-wrap">
    <img id="preview" src="" alt="Fingerprint preview"/>
  </div>

  <button class="btn" id="predictBtn" disabled onclick="predict()">
    <span id="btn-text">Analyse Fingerprint</span>
    <div class="spinner" id="spinner"></div>
  </button>

  <div id="error" class="error-msg"></div>

  <div id="result">
    <div class="result-label">Predicted Blood Group</div>
    <div class="blood-type" id="bloodType">—</div>
    <div class="confidence" id="confText"></div>
    <div class="bar-wrap" id="bars"></div>
  </div>
</div>

<script>
  const zone   = document.getElementById('zone');
  const input  = document.getElementById('fileInput');
  const btn    = document.getElementById('predictBtn');
  let   file   = null;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  input.addEventListener('change', () => { if (input.files[0]) handleFile(input.files[0]); });

  function handleFile(f) {
    file = f;
    const reader = new FileReader();
    reader.onload = e => {
      document.getElementById('preview').src = e.target.result;
      document.getElementById('preview-wrap').style.display = 'block';
    };
    reader.readAsDataURL(f);
    btn.disabled = false;
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').textContent = '';
  }

  async function predict() {
    if (!file) return;
    btn.disabled = true;
    document.getElementById('btn-text').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('error').textContent = '';
    document.getElementById('result').style.display = 'none';

    const form = new FormData();
    form.append('file', file);

    try {
      const res  = await fetch('/predict', { method:'POST', body: form });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      showResult(data);
    } catch(e) {
      document.getElementById('error').textContent = '⚠ ' + e.message;
    } finally {
      btn.disabled = false;
      document.getElementById('btn-text').style.display = 'block';
      document.getElementById('spinner').style.display = 'none';
    }
  }

  function showResult(data) {
    document.getElementById('bloodType').textContent = data.blood_group;
    document.getElementById('confText').textContent  =
      `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    const bars = document.getElementById('bars');
    bars.innerHTML = '';
    data.probabilities.forEach(([label, prob]) => {
      const pct = (prob * 100).toFixed(1);
      bars.innerHTML += `
        <div class="bar-row">
          <div class="bar-label">${label}</div>
          <div class="bar-bg"><div class="bar-fill" style="width:${pct}%"></div></div>
          <div class="bar-pct">${pct}%</div>
        </div>`;
    });

    document.getElementById('result').style.display = 'block';
  }
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train the model first (run train.py)."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    raw = request.files["file"].read()
    try:
        arr = preprocess(raw)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 400

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    blood_group = idx_to_class.get(top_idx, "Unknown")
    confidence  = float(preds[top_idx])

    probs_sorted = sorted(
        [(idx_to_class.get(i, str(i)), float(p)) for i, p in enumerate(preds)],
        key=lambda x: -x[1],
    )

    return jsonify({
        "blood_group":  blood_group,
        "confidence":   confidence,
        "probabilities": probs_sorted,
    })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
