# src/evaluate.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load the cleaned test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        "cleaned/Testing",
        image_size=(224,224),
        batch_size=32,
        label_mode="categorical",
        shuffle=False
    )

    # 2. Load the saved model
    model = load_model("brain_tumor_classifier.h5")

    # 3. Compute overall loss & accuracy
    loss, acc = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f}   Test accuracy: {acc:.4f}")

    # 4. Gather true labels & predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_pred = model.predict(test_ds)
    y_true_labels = y_true.argmax(axis=1)
    y_pred_labels = y_pred.argmax(axis=1)
    class_names = test_ds.class_names

    # 5. Classification report
    print(classification_report(
        y_true_labels, y_pred_labels, target_names=class_names
    ))

    # 6. Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
