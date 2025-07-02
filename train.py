# src/train.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import tensorflow as tf
from dataloader import load_datasets
import matplotlib.pyplot as plt


def build_model(num_classes: int):
    """Defines a CNN for 4 class classification."""
    return tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224,224,3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


if __name__ == "__main__":
    # 1. Load datasets
    train_ds, val_ds, class_names = load_datasets(
        data_dir="cleaned/Training",
        img_size=(224,224),
        batch_size=32,
        val_split=0.15,
        seed=42
    )
    num_classes = len(class_names)
    print("Classes:", class_names)

    # 2. Build & compile
    model = build_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=11
    )

    # 4. Plot and save training curves
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("training_curves.png")
    print("Saved training curves to training_curves.png")
    plt.show()

    # 5. Save the trained model
    model.save("brain_tumor_classifier.h5")
    print("Model saved to brain_tumor_classifier.h5")

