import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GlobalAveragePooling1D

X = np.load("X.npy")
y = np.load("y.npy")

model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=X.shape[1:]),
    Conv1D(64, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20, batch_size=16)

model.save("tcn_model.h5")