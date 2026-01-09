import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

import tensorflow as tf
from datetime import datetime
from preprocess import create_df, create_dataset
from sklearn.utils import class_weight
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.models import Model
from utils import plot_confusion_matrix, plot_training_curves, print_classification_report
from preprocess import index_label
import gc

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

version="DEMO"                                               #INCREMENT THIS WHEN YOU CHANGE THE CODE
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
experiment_name = f"mobilenetv2_{version}_{timestamp}"
save_dir = os.path.join("models", experiment_name)
os.makedirs(save_dir, exist_ok=True)
print(f"Experiment directory created: {save_dir}")

BATCH_SIZE = 16
IMG_SIZE = 224

# ------------------- Load datasets -------------------
train_df = create_df("data/train")
val_df = create_df("data/valid")
test_df = create_df("data/test")

print(f"\nDataset sizes: Train={len(train_df)}, Valid={len(val_df)}, Test={len(test_df)}")
for split_name, df in [("Train", train_df), ("Valid", val_df), ("Test", test_df)]:
    print(f"{split_name}: {dict(df['labels'].value_counts())}")

train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df, training=False)
test_ds = create_dataset(test_df, training=False)

for imgs, labels in train_ds.take(1):
    print("Train batch shapes:", imgs.shape, labels.shape) 
for imgs, labels in test_ds.take(1):
    print("Test batch shapes:", imgs.shape, labels.shape)


classes = np.unique(train_df['labels'])
weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_df['labels'])
class_weights = dict(zip(classes, weights))
class_weights_boosted = {
    0: class_weights[0] * 1.1,   
    1: class_weights[1] * 1.0,     
    2: class_weights[2] * 1.8
}
print(f"Boosted class weights: {class_weights_boosted}\n")

# ------------------- Build Model -------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    alpha=0.5
)
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax', dtype='float32')(x)

model = Model(inputs=inputs, outputs=output)
model.summary()

# ------------------- Stage 1: Train head -------------------
print("--- Training head layers ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks_head = [ tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss', restore_best_weights=True),]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights_boosted,
    callbacks=callbacks_head
)

# ------------------- Stage 2: Fine-tuning -------------------
print("\n--- Fine-tuning the model ---")
# Unfreeze the base model layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

bn_count = 0
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
        bn_count += 1
print(f"Frozen {bn_count} BatchNorm layers")

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-5, decay_steps=2000)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks_ft = [
    tf.keras.callbacks.EarlyStopping(patience=12, monitor='val_loss', mode='min', restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, "best_model.keras"), monitor='val_loss', mode='min', save_best_only=True),
]

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    initial_epoch=history_head.epoch[-1],
    class_weight=class_weights_boosted,
    callbacks=callbacks_ft
)

# ------------------- Evaluation -------------------
tf.keras.backend.clear_session()
gc.collect()

# Reload best model
best_model = tf.keras.models.load_model(
    os.path.join(save_dir, "best_model.keras"),
    compile=True
)

# Evaluate on test set
results = best_model.evaluate(test_ds)
print(f"\nTest Accuracy: {results[1] * 100:.2f}%, Test Loss: {results[0] * 100:.2f}%")

# Confusion matrix and report
class_names = [index_label[i] for i in range(len(index_label))]
print_classification_report(best_model, test_ds, class_names)

# Save plots
plot_training_curves(history_head, save_dir, title="Head Training")
plot_training_curves(history_ft, save_dir, title="Fine-tuning Training")

plot_confusion_matrix(
    best_model,
    test_ds,
    class_names,
    save_dir,
    normalize=True,
    title="Confusion Matrix"
)
