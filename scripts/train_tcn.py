"""
scripts/train_tcn.py
--------------------
Train the Temporal Convolutional Network on the prepared dataset.

HOW TO USE:
  python scripts/train_tcn.py

  Reads:  data/X.npy, data/y.npy   (produced by prepare_data.py)
  Writes: models/tcn_model.h5

ARCHITECTURE NOTE:
  Despite the name "TCN", this uses Conv1D layers — a simplified version.
  True TCN would also use dilated convolutions and residual connections,
  which is a great upgrade to add later.
  For a final-year project, Conv1D + GlobalAveragePooling is a solid start.
"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D

from guardia.config import DATA_DIR, MODELS_DIR, MODEL_PATH

# ── Load dataset ───────────────────────────────────────────────────────────
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
print(f"Loaded — X: {X.shape}  y: {y.shape}")

# ── Build model ────────────────────────────────────────────────────────────
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
    Conv1D(64, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(3, activation='softmax'),   # 3 classes: normal, fall, inactive
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# ── Train ──────────────────────────────────────────────────────────────────
model.fit(X, y, epochs=20, batch_size=16, validation_split=0.1)

# ── Save ───────────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
