"""
generate_demo_dataset.py
Generates a synthetic fingerprint-like dataset for smoke-testing train.py.
Run this ONLY if you don't have real fingerprint data yet.

Usage:
  python generate_demo_dataset.py --samples 200
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

BLOOD_GROUPS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


def make_fingerprint_like(seed: int) -> Image.Image:
    """Creates a synthetic ridge-pattern image resembling a fingerprint."""
    rng = np.random.default_rng(seed)
    img = np.zeros((128, 128), dtype=np.uint8)

    # Concentric elliptical ridges
    cx, cy = 64 + rng.integers(-8, 8), 64 + rng.integers(-8, 8)
    for r in range(5, 60, 7):
        for angle in np.linspace(0, 2 * np.pi, 360):
            x = int(cx + r * 1.1 * np.cos(angle))
            y = int(cy + r * 0.85 * np.sin(angle))
            if 0 <= x < 128 and 0 <= y < 128:
                img[y, x] = rng.integers(180, 255)

    pil = Image.fromarray(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=1.2))
    noise = Image.fromarray(rng.integers(0, 30, (128, 128), dtype=np.uint8))
    pil = Image.fromarray(np.clip(np.array(pil) + np.array(noise), 0, 255).astype(np.uint8))
    return pil


def generate(samples: int):
    print(f"Generating {samples} images per class …")
    for bg in BLOOD_GROUPS:
        folder = os.path.join("dataset", bg)
        os.makedirs(folder, exist_ok=True)
        for i in range(samples):
            seed = hash(bg + str(i)) % (2 ** 31)
            img = make_fingerprint_like(seed)
            img.save(os.path.join(folder, f"{bg.replace('+','p').replace('-','n')}_{i:04d}.png"))
    print("✅  Demo dataset created in ./dataset/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=200, help="Images per blood group")
    args = p.parse_args()
    generate(args.samples)
