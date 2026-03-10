import numpy as np
import sys
import os

TIME_STEPS = 30

def create_sequences(data, name):
    if len(data) <= TIME_STEPS:
        print(f"\n❌ ERROR: Your '{name}' dataset only has {len(data)} frames!")
        print(f"To use TCN, you must have more than {TIME_STEPS} frames (about 1 second of video).")
        print(f"👉 Please run:  python collect_data.py")
        print(f"👉 Enter '{name}' and let the camera record your movements for a few seconds, THEN press 'q'.\n")
        sys.exit(1)
        
    X = []
    for i in range(len(data) - TIME_STEPS):
        X.append(data[i:i+TIME_STEPS])
    return np.array(X)

try:
    normal = np.load("dataset/normal.npy")
    fall = np.load("dataset/fall.npy")
    inactive = np.load("dataset/inactive.npy")
except FileNotFoundError as e:
    print(f"\n❌ ERROR: Missing dataset file. {e}")
    print("Ensure you have run collect_data.py for 'normal', 'fall', and 'inactive'.\n")
    sys.exit(1)

X_normal = create_sequences(normal, "normal")
X_fall = create_sequences(fall, "fall")
X_inactive = create_sequences(inactive, "inactive")

X = np.concatenate([X_normal, X_fall, X_inactive])
y = np.concatenate([
    np.zeros(len(X_normal)),
    np.ones(len(X_fall)),
    np.full(len(X_inactive), 2)
])

np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset ready:", X.shape, y.shape)