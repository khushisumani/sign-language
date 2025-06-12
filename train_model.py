import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

df = pd.read_csv("sign_data.csv", dtype={"label": str})

X = df.drop("label", axis=1).values.astype(np.float32)
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

np.save("label_classes.npy", le.classes_)
print("Classes:", le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(len(le.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save("sign_model.h5")
print("Model trained and saved as sign_model.h5")
