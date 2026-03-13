import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

from guardia.config import DATA_DIR, MODELS_DIR, MODEL_PATH

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
print(f"Loaded — X: {X.shape}  y: {y.shape}")

# normalize per feature across all samples
n_samples, n_steps, n_features = X.shape
scaler = StandardScaler()
X_flat = scaler.fit_transform(X.reshape(-1, n_features))
X = X_flat.reshape(n_samples, n_steps, n_features)
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("Scaler saved")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes.astype(int), weights))
print("Class weights:", {["normal", "fall", "inactive"][k]: round(v, 2) for k, v in class_weight.items()})

model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", padding="causal", dilation_rate=1, input_shape=(n_steps, n_features)),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(64, kernel_size=3, activation="relu", padding="causal", dilation_rate=2),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(64, kernel_size=3, activation="relu", padding="causal", dilation_rate=4),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weight,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.1f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred, target_names=["normal", "fall", "inactive"]))

model.save(MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
