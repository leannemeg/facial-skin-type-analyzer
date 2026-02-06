# Skin Type Classifier — Simple Guide

This project trains a computer to recognize three skin types from photos: dry, normal, and oily. 

---

## Project Overview
- **Goal:** Classify face/skin images into one of three categories: dry, normal, oily.
- **Approach:** Use "transfer learning" with MobileNetV2, a lightweight, pre-trained neural network. We freeze most of the network first, train a small "head" on top, then fine-tune some deeper layers carefully.
- **Why Transfer Learning:** Instead of training from scratch (which needs lots of data), we reuse visual understanding learned from millions of images, then adapt it to our skin-type task.

---

## Repository Structure
- [data/](data/) — images split into `train`, `valid`, `test`, each with `dry/`, `normal/`, `oily/` folders.
- [scripts/preprocess.py](scripts/preprocess.py) — builds datasets and applies image preprocessing + augmentation.
- [scripts/train.py](scripts/train.py) — defines the model, trains it in 2 stages, saves results.
- [scripts/utils.py](scripts/utils.py) — plots training curves, confusion matrix, and prints metrics.
- [models/](models/) — each training run produces a timestamped folder with `best_model.keras` and plots.

---

## Prerequisites
- **OS:** Linux (tested)
- **Python:** 3.9+ recommended
- **GPU:** Optional but helpful; the code supports GPU training with TensorFlow.
- **Libraries:** TensorFlow/Keras, scikit-learn, pandas, matplotlib. Install as below.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install tensorflow keras scikit-learn pandas matplotlib
```

If you have an NVIDIA GPU, install GPU-enabled TensorFlow per TensorFlow’s docs.

---

## Data Setup
Your images should be organized like this:

```
data/
  train/
    dry/
    normal/
    oily/
  valid/
    dry/
    normal/
    oily/
  test/
    dry/
    normal/
    oily/
```

- **Train:** Used to teach the model.
- **Valid:** Used during training to check progress and prevent overfitting.
- **Test:** Held back until the end to measure true performance.

The code automatically assigns labels based on folder names:
- `dry` → 0
- `normal` → 1
- `oily` → 2

### Alternatives
- **K-fold cross-validation:** More robust when data is scarce; trains several models and averages metrics. Costs more compute.
- **Stratified splitting:** Maintain class proportions across splits if data is imbalanced.

---

## How Images Are Prepared
From [scripts/preprocess.py](scripts/preprocess.py):
- **Resize:** Every image is resized to 224×224 pixels.
- **Normalize:** Inputs are scaled using MobileNetV2’s `preprocess_input`.
- **Augmentation (training only):** Small random flips and brightness/contrast/saturation tweaks to help the model generalize and not memorize.
- **Batches:** Images are batched (16 at a time) and prefetched for speed.

This preparation happens automatically when you run training.

---

## How the Model Works
From [scripts/train.py](scripts/train.py):
- **Base:** MobileNetV2 (`include_top=False`, `alpha=0.5`) with ImageNet weights.
- **Head:** Global Average Pooling → Dense(128, ReLU, L2 regularization) → Dropout(0.5) → Dense(3, softmax).
- **Class Weights:** If classes are imbalanced, weights help the model pay more attention to underrepresented classes.

### Two-Stage Training
1. **Stage 1 — Train the head only**
   - Freeze the pre-trained base; train the new layers on top.
   - Optimizer: Adam with small learning rate (`1e-4`).
   - Early stopping monitors validation loss to avoid overfitting.

2. **Stage 2 — Fine-tune deeper layers**
   - Unfreeze some of the base (but keep BatchNorm layers frozen for stability).
   - Use a gentle, decaying learning rate (cosine decay around `1e-5`).
   - Save the best model (`best_model.keras`) based on validation loss.

This strategy is safer and more stable than unfreezing everything at once.

### Alternatives and Trade-offs
- **Unfreezing all layers at once:** Faster adaptation but riskier (can overfit, destabilize BN stats).
- **OneCycle or step decay LRs:** Also effective; cosine is simple and widely used for fine-tuning.
- **Focal loss:** Helpful for class imbalance; here class weights provide a simpler, effective remedy.

---

## How We Evaluate
From [scripts/utils.py](scripts/utils.py):
- **Accuracy & Loss Curves:** Plots for training and validation are saved in the run’s model folder.
- **Classification Report:** Precision, recall, F1-score per class.
- **Confusion Matrix:** Shows where predictions match/mismatch the true labels; normalized option available.

You’ll find outputs like:
- `head_training.png` and `fine-tuning_training.png` (names derived from titles)
- `confusion_matrix.png`
- `best_model.keras`

All inside a folder like:
```
models/mobilenetv2_best_model_a0.5-224_2025-12-01_13-47/
```

### What the Metrics Mean
- **Accuracy:** Fraction of correct predictions overall; can hide class imbalance issues.
- **Precision (per class):** Of items predicted as class X, how many are truly X?
- **Recall (per class):** Of true class X items, how many did we correctly find?
- **F1-score:** Harmonic mean of precision and recall; balances both.
- **Confusion matrix:** Grid of true vs. predicted classes; normalized entries show proportions, raw counts show volume.
- **Macro vs. Micro averages:** Macro averages metrics per class (good for imbalance); micro aggregates across all samples (good for overall performance).

---

## Run Training Yourself
Make sure your data is in `data/` as shown above, then:

```bash
# From the repository root
python -m scripts.train
```

What happens:
- The script creates a new folder under `models/` with a timestamp.
- It trains Stage 1, then Stage 2.
- It evaluates on the test set and saves plots + the best model.

---

## Use the Trained Model for Prediction
Here’s a simple snippet to predict the skin type for a single image using a saved model.

```python
import tensorflow as tf
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Update this path to your saved model
model = tf.keras.models.load_model('models/mobilenetv2_best_model_a0.5-224_2025-12-01_13-47/best_model.keras')

class_names = ['dry', 'normal', 'oily']
IMG_SIZE = 224

def load_and_preprocess(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype('float32')
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # batch of 1
    return arr

image_path = 'path/to/your/image.jpg'
inputs = load_and_preprocess(image_path)
preds = model.predict(inputs)
label_index = int(np.argmax(preds, axis=1)[0])
print('Predicted:', class_names[label_index])
```

---

## Troubleshooting Tips
- **Out of Memory:** Lower `BATCH_SIZE` in preprocessing or use smaller `IMG_SIZE`.
- **Mixed or incorrect classes:** Check your `data/` folder names and contents.
- **Slow training:** Prefetching and batching are already used; a GPU speeds things up.
