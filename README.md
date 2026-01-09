# Skin Type Classifier — Simple Guide

This project trains a computer to recognize three skin types from photos: dry, normal, and oily. 

---

## What You’ll Learn
- What the model does and how it was built
- How the data is organized and prepared
- How training works (in two stages) and why
- How we evaluate results (accuracy and confusion matrix)
- How to run training yourself and use the trained model
- Why these choices were made, alternatives considered, and key terms explained

---

## Project Overview
- **Goal:** Classify face/skin images into one of three categories: dry, normal, oily.
- **Approach:** Use "transfer learning" with MobileNetV2, a lightweight, pre-trained neural network. We freeze most of the network first, train a small "head" on top, then fine-tune some deeper layers carefully.
- **Why Transfer Learning:** Instead of training from scratch (which needs lots of data), we reuse visual understanding learned from millions of images, then adapt it to our skin-type task.

### Why This Approach
- **Data efficiency:** Transfer learning shines when labeled data is limited. Pretrained features already capture edges, textures, and colors relevant to skin appearance.
- **Compute constraints:** MobileNetV2 is designed to be fast and memory-friendly, ideal for laptops and modest GPUs.
- **Stability:** Two-stage training (head-first, then careful fine-tuning) reduces the risk of "catastrophic forgetting" where the model loses useful pretrained features.
- **Practicality:** The pipeline balances accuracy, speed, and simplicity so you can iterate quickly.

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

### Why This Split (Train/Valid/Test)
- **Train vs. Valid:** The validation set provides an impartial checkpoint to tune the model (e.g., early stopping) without contaminating the final evaluation.
- **Test:** Kept untouched to estimate generalization to new, unseen data.
- **Avoiding leakage:** Ensures similar images (or near-duplicates) don’t appear across splits, which would inflate metrics.

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

### Why These Choices
- **224×224 resolution:** Matches common pretrained backbones and offers a good balance of detail vs. speed/memory.
- **MobileNetV2 `preprocess_input`:** Aligns input scaling with how the backbone was originally trained (typically mapping pixel values to a range suitable for the network), improving convergence.
- **Subtle augmentation:** Skin classification is sensitive to color/texture. Gentle changes improve robustness without distorting signals (e.g., oiliness shine) that the model needs.
- **Batch size 16:** Practical default for typical GPU/CPU memory; tune up/down based on your hardware.

### Augmentation Terms (What They Mean)
- **Flip:** Horizontal mirroring to handle left/right orientation differences.
- **Brightness/Contrast/Saturation:** Photometric tweaks that simulate lighting/camera variability.
- **Normalization:** Transforming pixel values to the scale expected by the model to stabilize training.

---

## How the Model Works
From [scripts/train.py](scripts/train.py):
- **Base:** MobileNetV2 (`include_top=False`, `alpha=0.5`) with ImageNet weights.
- **Head:** Global Average Pooling → Dense(128, ReLU, L2 regularization) → Dropout(0.5) → Dense(3, softmax).
- **Class Weights:** If classes are imbalanced, weights help the model pay more attention to underrepresented classes.

### Why These Layers and Values
- **Global Average Pooling (GAP):** Converts spatial feature maps into compact vectors without adding many parameters, reducing overfitting.
- **Dense(128) with ReLU + L2:** A modest-sized layer to adapt pretrained features to our 3-class task. L2 (weight decay) discourages overly large weights, improving generalization.
- **Dropout(0.5):** Randomly zeroes half the activations during training, a strong regularizer that curbs co-adaptation.
- **Dense(3, softmax):** Produces a probability distribution over the three classes.
- **`alpha=0.5`:** Controls MobileNetV2 width (fewer channels). Smaller `alpha` reduces compute/memory while often retaining enough capacity for a 3-class task.

### Class Weights (How/Why)
- **Why:** When one class has fewer samples, the model can ignore it. Class weights increase the loss contribution for rare classes.
- **Typical formula:** `weight_c = N / (C * n_c)` where `N` is total samples, `C` number of classes, and `n_c` count for class `c`.
- **Trade-offs:** Helps recall on rare classes but can slightly reduce precision on common classes. Alternative: focal loss.

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

### Why Two Stages (Deeper Rationale)
- **Stability first:** Training only the head lets the classifier align with fixed features without disrupting well-formed pretrained representations.
- **Controlled adaptation:** Fine-tuning selectively (and freezing BatchNorm) reduces noisy updates that can destabilize distributions learned on ImageNet.
- **Learning rate schedule:** Cosine decay starts small and shrinks further, encouraging fine-grained adjustments instead of drastic changes.

### BatchNorm (What/Why)
- **What:** Normalizes activations per batch, tracking running means/variances.
- **Why Freeze in fine-tuning:** Small batch sizes and domain shift can yield unreliable statistics. Freezing preserves stable behavior while adjusting other weights.

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

### Tips While Training
- **Monitor validation loss:** If it plateaus or worsens, early stopping prevents overfitting.
- **Adjust batch size:** If you hit memory limits, reduce it; if training is slow and memory allows, increase it.
- **Tweak augmentation:** If the model struggles with lighting changes, increase brightness/contrast randomization slightly; if color is critical, keep augmentations conservative.

### Changing Image Size or Model Width
- In [scripts/train.py](scripts/train.py), the key settings are:
  - `IMG_SIZE = 224`
  - `alpha=0.5` in `MobileNetV2(...)`
- Larger images or a larger `alpha` may improve accuracy but cost more compute.

### Why 224 and `alpha=0.5` (Trade-offs)
- **Resolution:** Higher than 224 can capture finer skin textures (pores, shine), but increases memory and latency. Lower than 224 is faster but may lose subtle cues.
- **Width (`alpha`):** Wider models (`alpha` closer to 1.0) have more capacity and can improve accuracy on complex data; narrower models run faster with less overfitting risk on small datasets.
- **Rule of thumb:** Increase image size before width if faces are small in the frame; increase width before image size if textures are already clear but patterns seem underfit.

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

### Interpreting Predictions
- **Softmax probabilities:** The output vector shows confidence per class; the highest value is the predicted class.
- **Confidence vs. correctness:** High confidence can still be wrong if the image is out-of-distribution (e.g., unusual lighting or makeup).
- **Batch inference:** For multiple images, stack them into a batch to speed up prediction.

---

## Common Questions
- **Do I need a GPU?** No, but it helps. The code runs on CPU; training just takes longer.
- **How much data is enough?** More is better. If one class has far fewer images, results may be skewed. Class weights help but can’t replace data.
- **Why freeze BatchNorm layers in fine-tuning?** It stabilizes training when reusing pre-trained features.
- **What if validation results get worse?** Early stopping will prevent wasting time; consider adjusting learning rate, data balance, or augmentation.

### Why Not Other Architectures?
- **EfficientNet:** Strong accuracy and parameter efficiency; heavier than MobileNetV2 in some variants. Good upgrade if you have more compute.
- **ResNet50/101:** Robust, widely used; heavier and slower on CPUs, may overfit small datasets without strong regularization.
- **Vision Transformers (ViT):** Powerful with large data; typically need more data or stronger augmentation and longer training.
- **ConvNeXt/MobileNetV3:** Worth trying for incremental gains; complexity increases setup and tuning effort.

### Why Not Train From Scratch?
- Requires substantial labeled data and compute; transfer learning reaches good accuracy faster with less risk of overfitting.

---

## Troubleshooting Tips
- **Out of Memory:** Lower `BATCH_SIZE` in preprocessing or use smaller `IMG_SIZE`.
- **Mixed or incorrect classes:** Check your `data/` folder names and contents.
- **Slow training:** Prefetching and batching are already used; a GPU speeds things up.

### Additional Remedies
- **Imbalance persists:** Try focal loss or oversampling the minority class in addition to class weights.
- **Overfitting:** Increase dropout, L2 regularization, or augmentation; consider reducing the Dense layer size.
- **Underfitting:** Increase `alpha` or image size; unfreeze more layers during fine-tuning; train a bit longer with a carefully managed LR.
- **Color shifts:** If augmentation harms color-based cues, reduce saturation/brightness changes.
- **Reproducibility:** Set random seeds and pin library versions to minimize variance between runs.

---

## Credits and References
- MobileNetV2: Sandler et al., 2018 (efficient vision models)
- Keras/TensorFlow for modeling and training
- scikit-learn for metrics

If you’d like help customizing the README for your dataset or adding an inference script, let me know.

---

## Terminology Cheat Sheet
- **Transfer learning:** Reusing a model trained on a large dataset for a new, related task.
- **Fine-tuning:** Unfreezing some pretrained layers and training them on your data to adapt features.
- **BatchNorm (BN):** A layer that normalizes activations using batch statistics; improves training stability.
- **Global Average Pooling (GAP):** Aggregates each feature map by averaging across spatial dimensions.
- **Dropout:** Regularization that randomly disables activations during training to reduce overfitting.
- **L2 regularization (weight decay):** Penalizes large weights to encourage simpler models.
- **Softmax:** Converts logits into probabilities that sum to 1 across classes.
- **Cosine decay LR:** Learning rate schedule that smoothly decays following a cosine curve.
- **Early stopping:** Halts training when validation performance stops improving.
- **Class weights:** Adjusts the loss contribution per class to address imbalance.
- **Confusion matrix:** Table showing true vs. predicted class counts or proportions.

## Reproducibility Notes
- **Seeds:** Set seeds for Python, NumPy, and TensorFlow to reduce run-to-run variance.
- **Determinism:** Exact determinism can be hard with GPU kernels; aim for stable averages.
- **Version pinning:** Use a `requirements.txt` or pinned versions to keep environments consistent.
