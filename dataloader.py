# src/dataloader.py

import tensorflow as tf
import argparse

def load_datasets(
    data_dir: str,
    img_size=(224,224),
    batch_size=32,
    val_split=0.15,
    seed=42
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="cleaned/Training")
    args = parser.parse_args()

    train_ds, val_ds, classes = load_datasets(args.data_dir)
    print("Classes:", classes)
    print("Train batches:", len(train_ds), " Val batches:", len(val_ds))
    for x, y in train_ds.take(1):
        print("Images batch shape:", x.shape, "Labels batch shape:", y.shape)

