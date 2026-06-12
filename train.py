"""
Blood Group Prediction from Fingerprint Images using CNN + TensorFlow
=====================================================================
Dataset folder structure expected:
  dataset/
    A+/   *.bmp or *.png fingerprint images
    A-/
    B+/
    B-/
    AB+/
    AB-/
    O+/
    O-/

Usage:
  python train.py --data_dir dataset --epochs 30 --batch_size 32
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE   = (128, 128)
NUM_CLASSES = 8
BLOOD_GROUPS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


# ── Argument Parsing ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="dataset",    help="Root dataset directory")
    p.add_argument("--model_out",  default="models/bloodgroup_cnn.h5")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_split",  type=float, default=0.2)
    p.add_argument("--lr",         type=float, default=1e-3)
    return p.parse_args()


# ── Data Pipeline ─────────────────────────────────────────────────────────────
def build_generators(data_dir, batch_size, val_split):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=val_split,
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


# ── CNN Model ─────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*IMG_SIZE, 1), name="fingerprint")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="blood_group")(x)

    return keras.Model(inputs, outputs, name="BloodGroupCNN")


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print("\n📁  Loading data from:", args.data_dir)
    train_gen, val_gen = build_generators(args.data_dir, args.batch_size, args.val_split)

    # Save class index map
    class_indices = train_gen.class_indices
    idx_to_class  = {v: k for k, v in class_indices.items()}
    with open("models/class_indices.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print("📌  Classes:", class_indices)

    model = build_model(NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.model_out, save_best_only=True, monitor="val_accuracy", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    print("\n🚀  Training started …\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_history(history)
    _plot_confusion(model, val_gen, idx_to_class)

    print(f"\n✅  Model saved to {args.model_out}")
    return model, history


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("models/training_history.png", dpi=120)
    plt.close()
    print("📊  Training plot saved to models/training_history.png")


def _plot_confusion(model, val_gen, idx_to_class):
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    y_true = val_gen.classes
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title("Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=120)
    plt.close()
    print("📊  Confusion matrix saved to models/confusion_matrix.png")
    print("\n📋  Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=labels))


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
