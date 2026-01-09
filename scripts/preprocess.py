import os
import pandas as pd
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input

label_index = {"dry": 0, "normal": 1, "oily": 2}
index_label = {v: k for k, v in label_index.items()}

IMG_SIZE = 224
BATCH_SIZE = 16

def create_df(base):
    dd = {"images": [], "labels": []}
    for label_name in os.listdir(base):
        label_path = os.path.join(base, label_name)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            img_path = os.path.join(label_path, fname)
            dd["images"].append(img_path)
            dd["labels"].append(label_index[label_name])
    return pd.DataFrame(dd)


def preprocess_image(img_path, label, training=True):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1) 
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.adjust_contrast(img, tf.random.uniform([], 0.8, 1.2))
    
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    
    return img, label


def create_dataset(df, training=True):
    images = df['images'].values
    labels = df['labels'].values
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    ds = ds.map(lambda x, y: preprocess_image(x, y, training),
                num_parallel_calls=4)

    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 300), reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(2)
    return ds
