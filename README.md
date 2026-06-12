# 🩸 Blood Group Prediction from Fingerprints — CNN + TensorFlow

Predict ABO+Rh blood groups (A+, A−, B+, B−, AB+, AB−, O+, O−) from
fingerprint images using a custom Convolutional Neural Network built with
TensorFlow / Keras, with a Flask web UI for inference.

---

## Project Structure

```
bloodgroup/
├── train.py                  # CNN model training script
├── app.py                    # Flask web app for inference
├── generate_demo_dataset.py  # Synthetic data generator (for testing)
├── requirements.txt
├── dataset/                  # Your fingerprint images go here
│   ├── A+/
│   ├── A-/
│   ├── B+/
│   ├── B-/
│   ├── AB+/
│   ├── AB-/
│   ├── O+/
│   └── O-/
└── models/                   # Auto-created after training
    ├── bloodgroup_cnn.h5
    └── class_indices.json
```

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Prepare Dataset

### Option A — Real Dataset
Use any publicly available fingerprint dataset (e.g., NIST SD-4, Sokoto
Coventry Fingerprint Dataset) with images organised into the 8 blood-group
subfolders shown above.

### Option B — Demo / Smoke-Test Data
```bash
python generate_demo_dataset.py --samples 200
```
This creates synthetic ridge patterns (~200 images per class). Accuracy will
be low (random patterns), but it lets you verify the full pipeline works.

---

## 3. Train the CNN

```bash
python train.py \
  --data_dir  dataset \
  --epochs    30 \
  --batch_size 32
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `dataset` | Root folder with 8 class sub-dirs |
| `--model_out` | `models/bloodgroup_cnn.h5` | Where to save the trained model |
| `--epochs` | `30` | Max training epochs |
| `--batch_size` | `32` | Images per batch |
| `--val_split` | `0.2` | Fraction held out for validation |
| `--lr` | `0.001` | Initial learning rate |

After training you will find:
- `models/bloodgroup_cnn.h5` — saved model  
- `models/class_indices.json` — label mapping  
- `models/training_history.png` — accuracy / loss curves  
- `models/confusion_matrix.png` — per-class confusion matrix  

---

## 4. Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser, upload a fingerprint image,
and click **Analyse Fingerprint** to get the prediction with confidence bars
for all 8 blood groups.

---

## CNN Architecture

```
Input (128×128 grayscale)
  └─ Conv2D(32) → BN → MaxPool
  └─ Conv2D(64) → BN → MaxPool
  └─ Conv2D(128) → BN → MaxPool
  └─ Conv2D(256) → BN → GlobalAvgPool
  └─ Dense(512) → Dropout(0.4)
  └─ Dense(256) → Dropout(0.3)
  └─ Dense(8, softmax)    ← blood group probabilities
```

Regularisation: Batch Normalisation + Dropout + data augmentation
(rotation, shift, zoom, horizontal flip).

---

## Tips for Better Accuracy

- Use at least **500–1000 real fingerprint images per class**.
- Increase `--epochs` to 50–100 with early stopping (already included).
- Fine-tune a pretrained backbone (e.g., MobileNetV2) for small datasets:
  replace the Conv blocks in `build_model()` with `keras.applications.MobileNetV2`.
- Ensure consistent image quality (resolution, lighting, sensor type).

---

## License
MIT
