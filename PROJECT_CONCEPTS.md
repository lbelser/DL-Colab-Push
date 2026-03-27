# Deep Learning Project — Concept Guide

> Written for a student with a data science / machine learning background but no prior neural network or image classification experience. All explanations are grounded in **this specific project**: classifying paintings by their artist (23 classes, ~13 000 WikiArt images).

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Image Dimensions and Why They Matter](#2-image-dimensions-and-why-they-matter)
3. [Pixel Statistics](#3-pixel-statistics)
4. [How Neural Network Training Works](#4-how-neural-network-training-works)
5. [What Is an Epoch?](#5-what-is-an-epoch)
6. [Convolutional Neural Networks (CNNs)](#6-convolutional-neural-networks-cnns)
7. [Data Augmentation](#7-data-augmentation)
8. [Pre-trained Networks vs. Training from Scratch](#8-pre-trained-networks-vs-training-from-scratch)
9. [Frozen Base — What It Means and Why We Use It](#9-frozen-base--what-it-means-and-why-we-use-it)
10. [Ways to Tweak a Pre-trained Network](#10-ways-to-tweak-a-pre-trained-network)
11. [Training vs. Validation Accuracy Curves](#11-training-vs-validation-accuracy-curves)
12. [Models in This Project and Their Results](#12-models-in-this-project-and-their-results)
13. [How to Improve Further — New Models and Advanced Techniques](#13-how-to-improve-further--new-models-and-advanced-techniques)
14. [Should We Use an Ensemble of More Than Two Models?](#14-should-we-use-an-ensemble-of-more-than-two-models)
15. [Quick Reference Glossary](#15-quick-reference-glossary)

---

## 1. The Problem We Are Solving

We are doing **image classification**: given a painting as input, predict which of 23 famous artists painted it.

This is a *multi-class classification* problem — the same concept you know from ML (e.g., classifying species from iris measurements), but now the input is an image instead of a row in a spreadsheet.

**The challenge** is that artists share stylistic elements (many are Impressionists), the dataset is unbalanced (Van Gogh has 1 322 paintings; Salvador Dalí has only 336), and artistic style is abstract — you cannot write explicit rules for "looks like Van Gogh". Neural networks learn those rules automatically from examples.

---

## 2. Image Dimensions and Why They Matter

### What an image is to a neural network

A digital image is just a 3-dimensional array of numbers:

```
(height, width, channels)
```

A 224 × 224 RGB image is a tensor of shape `(224, 224, 3)` — 224 rows of pixels, 224 columns, and 3 colour channels (Red, Green, Blue). Each value is a number between 0 and 255 (or 0 and 1 after normalisation).

### Why we resize everything to 224 × 224

Neural networks require a **fixed input size** — every image fed into the network must have the same shape, because the weight matrices inside the network have fixed dimensions.

The original WikiArt images are 512 × 512 pixels. We resize them to **224 × 224** for two reasons:

1. **Pre-trained model compatibility** — ResNet50 and EfficientNet were trained on ImageNet at 224 × 224. Using the same size lets us reuse their learned filters directly.
2. **Computational cost** — A 224 × 224 image has 3× fewer pixels than 512 × 512. Training is much faster and requires less GPU memory.

### The trade-off

Resizing discards fine detail. For paintings, high-resolution brushstroke texture can be a style signal (pointillism, impasto). At 224 × 224, some of this is lost. This is a known limitation of the current pipeline.

### Normalisation

Pixel values are divided by 255, converting the range from `[0, 255]` to `[0, 1]`. This is the same reason you scale numerical features before training a logistic regression — gradient-based optimisation converges much faster on small, uniform-scale numbers.

---

## 3. Pixel Statistics

### What they are

Pixel statistics are simple descriptive statistics (mean, standard deviation) computed per colour channel across the entire dataset:

| Channel | Mean  | Std   |
|---------|-------|-------|
| Red     | 0.536 | 0.265 |
| Green   | 0.491 | 0.259 |
| Blue    | 0.431 | 0.262 |

### Why we compute them

When using a **pre-trained model** (e.g., ResNet50 trained on ImageNet), the model's internal filters were calibrated for data with a specific colour distribution — the ImageNet mean and standard deviation:

| Channel | ImageNet Mean | ImageNet Std |
|---------|---------------|--------------|
| Red     | 0.485         | 0.229        |
| Green   | 0.456         | 0.224        |
| Blue    | 0.406         | 0.225        |

Think of this like a z-score normalisation from statistics: the pre-trained model "expects" inputs to look like ImageNet. If our data has a very different distribution, the model's learned features won't apply as well.

### What we found

Our painting dataset has **higher means and higher standard deviations** than ImageNet. This makes sense: paintings have vibrant, diverse colours and more varied lighting than photos of everyday objects. However, the difference is small enough that using the ImageNet preprocessing function (which applies the correct mean/std rescaling) worked without custom normalisation.

**In short:** pixel statistics tell us whether our data distribution matches what the pre-trained model expects. They help us decide whether to apply custom normalisation.

---

## 4. How Neural Network Training Works

This is the most important concept. You already know the idea of a model learning from data — here is how it works mechanically.

### Step 1 — Forward Pass

An image goes through the network layer by layer. Each layer applies a mathematical transformation (matrix multiplications, applying an activation function). The final layer outputs a vector of 23 numbers — one probability per artist — called the **softmax output**.

```
Image (224×224×3) → Layer 1 → Layer 2 → ... → 23 probabilities
```

### Step 2 — Loss Function

The **loss function** measures how wrong the prediction is. We use *cross-entropy loss*, which compares the predicted probability distribution to the true label:

- If the model predicted 90% chance "Van Gogh" and the answer was Van Gogh → low loss.
- If the model predicted 2% chance "Van Gogh" and the answer was Van Gogh → high loss.

This is directly analogous to log-loss in your ML courses.

### Step 3 — Backpropagation

This is the key step unique to neural networks. The network calculates the **gradient** of the loss with respect to every single weight in the network. This tells us: "if I increase this weight slightly, does the loss go up or down?"

Backpropagation is an efficient application of the chain rule from calculus, applied layer by layer from the output backwards to the input (hence "back" propagation).

### Step 4 — Gradient Descent Update

Every weight is nudged in the direction that reduces the loss:

```
new_weight = old_weight - learning_rate × gradient
```

The **learning rate** controls the step size. Too large → overshoots optimal values. Too small → training is very slow. We use the **Adam optimiser**, which adaptively adjusts the learning rate for each weight — it's the most common choice in deep learning.

### Step 5 — Repeat

We repeat steps 1–4 for every mini-batch of 32 images, updating weights continuously. After the model has seen every image in the training set once, one **epoch** is complete.

### Analogy

Think of a sculptor working from a blurry reference photo. Each pass, the sculptor makes small adjustments (reduces the loss). Backpropagation is the sculptor's critical eye that tells them exactly where to chisel.

---

## 5. What Is an Epoch?

An **epoch** is one complete pass through the entire training dataset.

With ~9 200 training images and a batch size of 32:
- One epoch = 9200 / 32 ≈ **288 gradient update steps**
- After 30 epochs, the model has seen each image 30 times (each time potentially augmented differently)

Epochs matter because:
- Too few → model hasn't converged yet (underfitting)
- Too many → model memorises the training data (overfitting)

We use **EarlyStopping** (a callback) to automatically stop training when the validation accuracy stops improving for 5 consecutive epochs, preventing overfitting.

---

## 6. Convolutional Neural Networks (CNNs)

Before transfer learning makes sense, you need to know what a CNN actually learns.

### Regular (Dense) vs. Convolutional Layers

A regular dense layer connects every neuron to every input. For a 224 × 224 × 3 image (150 528 values) this would require hundreds of millions of weights just in the first layer, and it would treat each pixel as independent — ignoring spatial structure.

A **convolutional layer** instead slides a small filter (e.g., 3×3 pixels) across the image, computing a dot product at each position. This:
- Is much more parameter-efficient (a 3×3 filter has only 9 weights)
- Preserves spatial relationships
- Detects the same pattern regardless of where it appears in the image

### What Each Layer Learns

This is crucial for understanding transfer learning:

| Network Depth | What Is Learned |
|---------------|-----------------|
| First layers | Edges, colour gradients, corners |
| Middle layers | Textures, shapes, brushstroke patterns |
| Deeper layers | Semantic features: "this looks like an Impressionist brushstroke" |
| Final layer | Class-specific combinations of high-level features |

Early layers are generic (the same edges appear in paintings and photos). Deep layers are domain-specific.

### Our Baseline CNN

Three convolutional blocks followed by a classification head:

```
Conv(32 filters) → BatchNorm → ReLU → MaxPool   ← detects edges
Conv(64 filters) → BatchNorm → ReLU → MaxPool   ← detects textures
Conv(128 filters) → BatchNorm → ReLU → MaxPool  ← detects brushstrokes
GlobalAveragePool → Dropout → Dense(256) → Dense(23, softmax)
```

**BatchNormalization** normalises the outputs of a layer during training — it stabilises and speeds up learning (analogous to feature scaling in classic ML).

**MaxPooling** reduces spatial size by keeping only the maximum value in each 2×2 region — it compresses information and makes the representation position-invariant.

**GlobalAveragePooling** collapses each feature map to a single number (its mean), drastically reducing the number of parameters before the final dense layers.

**Dropout** randomly sets a fraction of outputs to zero during training — it is a regularisation technique that prevents overfitting (forces the network not to rely too heavily on any single neuron).

---

## 7. Data Augmentation

### What It Is

Data augmentation is the practice of **artificially creating new training examples by randomly transforming existing images** in ways that preserve the label.

For example, a horizontal flip of a Monet painting is still a Monet painting. A slightly brighter version of a Van Gogh is still a Van Gogh. By applying random transformations during training (different each epoch), we effectively expand our dataset and force the model to learn features that are invariant to these distortions.

### Why It Is Critical Here

1. **Limited data**: ~9 200 training images for 23 classes (~400 per class on average). Deep networks need much more data to generalise well.
2. **Overfitting**: Without augmentation, the model memorises the exact training images rather than learning general style features.
3. **Digitalisation variability**: Paintings are photographed or scanned under different lighting, angles, and colour calibrations. Augmentation simulates this variability.

### What We Used

**Phase 1 — Basic Augmentation:**
- Random horizontal flip (paintings have no inherent orientation)
- Small random brightness shift (±10%)
- Small random contrast shift (±10%)

**Phase 2 — Strong Augmentation** (added after observing a 11-point train-validation gap):
- All basic augmentations, plus:
- Random saturation shift (±20%)
- Very small random hue shift (±2%)
- Random crop (85–100% of image) then resize back to 224 × 224

### An Important Finding from This Project

Strong augmentation **did not help ResNet50**. Its validation accuracy stayed at 78.8% while the training-validation gap actually *increased*. The likely reason: **composition is a style signal in art**. If Cézanne always places the focal point in the top third, a random crop destroys that signal. This is a domain-specific caveat — augmentations that work for natural photos may harm art classification.

---

## 8. Pre-trained Networks vs. Training from Scratch

### Training from Scratch

You randomly initialise all weights and train the entire network on your data. This requires:
- Large datasets (millions of images)
- Long training time
- High risk of overfitting on small datasets

Our baseline CNN was trained from scratch on ~9 200 images. Result: **48.2% accuracy**.

### Transfer Learning (Pre-trained Networks)

A network like ResNet50 was already trained on **ImageNet** — 1.2 million images across 1 000 categories (dogs, cars, furniture, etc.). After training, its convolutional layers contain rich visual feature detectors built from enormous amounts of data.

The key insight: **visual features are reusable**. The edge detectors, texture detectors, and shape detectors that help classify dogs and cars also help classify paintings. We don't need to re-learn them from scratch.

Transfer learning means:
1. Take the pre-trained convolutional backbone (all layers except the final classification layer)
2. Replace the final layer with a new one for our 23-class problem
3. Fine-tune (or just train the new layer) on our smaller dataset

Result with ResNet50: **78.8% accuracy** — a 30+ point improvement over training from scratch.

### The Analogy

Transfer learning is like hiring an experienced photographer to learn art appraisal. They already know how to see (edges, textures, composition). You just need to teach them the *art-specific* vocabulary, not vision from zero.

---

## 9. Frozen Base — What It Means and Why We Use It

### What "Frozen" Means

When we say the base model is **frozen**, we mean its weights are **not updated during training** — they stay exactly as they were after ImageNet training. Only the new classification head (Dense layers we added) is trained.

In code: `base_model.trainable = False`

### Why Freeze in Phase 1?

1. **Stability**: The pre-trained features are high quality. Immediately fine-tuning all 25 million weights with a randomly-initialised head would produce large, chaotic gradients that could destroy the learned features ("catastrophic forgetting").
2. **Speed**: Fewer parameters to update → much faster training.
3. **Data efficiency**: With only ~9 200 images, training 25 million weights would severely overfit.

### The Two-Phase Strategy Used Here

**Phase 1 — Frozen base (30 epochs, LR = 1e-4):**
- Freeze the entire ResNet50/EfficientNet backbone
- Only train the new Dense head (256 neurons → 23 classes)
- Goal: teach the head to use the pre-trained features for our problem

**Phase 2 — Fine-tuning (15 epochs, LR = 1e-5):**
- Unfreeze layers from index ~140 onward (the deeper, more task-specific layers)
- Use a **10× smaller learning rate** (1e-5 instead of 1e-4) to make tiny adjustments
- Goal: slightly adapt the deep features to paintings, without forgetting what they learned on ImageNet

---

## 10. Ways to Tweak a Pre-trained Network

There are several strategies, ranging from minimal to aggressive modification:

### 1. Feature Extraction (Fully Frozen)
Freeze everything, only train the new classification head. Fastest, safest, but the features are fixed to ImageNet patterns.

**Used in Phase 1 of this project.**

### 2. Partial Fine-tuning (Partially Frozen)
Freeze early layers (generic edge/texture detectors), unfreeze later layers (high-level, domain-specific features). Use a low learning rate.

**Used in Phase 2 of this project** — unfreeze from layer 140 onward in ResNet50.

### 3. Full Fine-tuning
Unfreeze everything and train end-to-end with a low learning rate. Risky without a large dataset (can catastrophically forget ImageNet features). Best when your target domain is very different from ImageNet.

**Not used here** — with ~9 200 images and 25 million parameters, this would almost certainly overfit badly.

### 4. Layer-wise Learning Rate Decay
Assign different learning rates to different layers — very small LR for early layers, slightly larger for late layers. More precise than binary freeze/unfreeze. Commonly used in advanced fine-tuning.

**Not used here, but would make sense.**

### 5. Custom Classification Head (Deeper Head)
Instead of a single `Dense(256) → Dense(23)` head, add more layers, attention mechanisms, or skip connections. This gives the model more capacity to combine the backbone's features.

**Our head is fairly minimal. A deeper head could help.**

### 6. Domain-Adaptive Batch Normalisation
Unfreeze only BatchNorm layers while keeping convolutional weights frozen. This re-calibrates the normalisation statistics for your domain with very low risk of overfitting.

**Not used here, but is a well-regarded technique.**

### 7. Label Smoothing
Instead of training the model to predict probability 1.0 for the correct class, we use 0.9 (the remaining 0.1 is spread evenly across other classes). This prevents the model from becoming overconfident and improves generalisation.

**Used in Phase 2** with `label_smoothing=0.1`.

### Summary of What Was and Was Not Used

| Technique | Used? |
|-----------|-------|
| Feature extraction (frozen base) | Yes — Phase 1 |
| Partial fine-tuning | Yes — Phase 2 |
| Full fine-tuning | No |
| Layer-wise learning rates | No — would be a natural next step |
| Deeper custom head | No — minimal head used |
| Domain-adaptive BatchNorm | No — worth trying |
| Label smoothing | Yes — Phase 2 |

---

## 11. Training vs. Validation Accuracy Curves

### Why Tracking Both Matters

- **Training accuracy**: how well the model fits the examples it was trained on
- **Validation accuracy**: how well the model generalises to unseen data

The gap between them is the key diagnostic signal.

### Why the Training Line Is Smooth, Validation Is Not (Baseline CNN)

**Training accuracy is smooth** because:
- It is averaged over the full training set (~9 200 images / 32 per batch = 288 steps per epoch)
- Each step's noise averages out over 288 updates
- The metric you see per epoch is a rolling average over many batches

**Validation accuracy is noisy** because:
- The validation set is smaller (~1 968 images = ~62 batches)
- There is no averaging over many batches — it is a single snapshot at the end of each epoch
- A few difficult batches can swing the number significantly
- The model's weights can oscillate slightly between epochs, making val accuracy jump up and down

### Why Both Are Smooth for ResNet50

ResNet50 starts from **excellent pre-trained features**. It does not need many epochs to settle — the loss landscape is much smoother and more convex near the pre-trained weight solution. The model converges confidently without oscillation, so both training and validation accuracy rise smoothly and consistently.

The baseline CNN, starting from random weights, has to search a much rougher loss landscape — it takes unstable, exploratory steps before finding a stable region.

### Reading the Gap

| Pattern | Diagnosis |
|---------|-----------|
| Train ≈ Val (both low) | Underfitting — model too simple or too few epochs |
| Train >> Val | Overfitting — model memorises training data |
| Train ≈ Val (both high) | Good generalisation |
| Val > Train (early epochs) | Normal with dropout/augmentation (dropout is off during validation) |

In our project: ResNet50 Phase 2 had ~92% train vs ~81% validation (11-point gap) — moderate overfitting that augmentation could not fully resolve.

---

## 12. Models in This Project and Their Results

| Model | Architecture | Approach | Test Accuracy |
|-------|-------------|----------|---------------|
| Baseline CNN | 3-layer conv net | From scratch | 48.2% |
| ResNet50 | 50-layer residual net | Transfer + fine-tune | 78.8% |
| EfficientNetB0 | Scaled efficient net | Transfer + fine-tune | 73.9% |
| EfficientNetB2 | Larger EfficientNet | Transfer + fine-tune | 76.1% |
| **Ensemble** | ResNet50 + EfficientNetB2 | Average predictions | **80.5%** |

The ensemble simply averages the softmax probability outputs of two models. When one model is uncertain, the other may be more confident — averaging reduces overall error.

---

## 13. How to Improve Further — New Models and Advanced Techniques

### Recommended New Models

#### Option A — Vision Transformer (ViT)
**What it is:** Instead of convolutional filters, ViT splits the image into small patches (e.g., 16×16 pixels) and processes them as a sequence using a Transformer — the same architecture that powers language models like GPT and BERT.

**Why it makes sense here:**
- Transformers model long-range dependencies (the relationship between a brushstroke in the top-left and a colour choice in the bottom-right)
- Convolutions only see local neighbourhoods
- Art style often involves **global composition** — ViT can capture this better
- Pre-trained ViT-B/16 on ImageNet-21k achieves state-of-the-art on fine-grained classification tasks

**Keras usage:** `keras.applications.VisionTransformer` (or via `timm` / `HuggingFace`)

**Expected benefit:** The fundamentally different feature representation from ResNet/EfficientNet makes it an excellent ensemble partner.

#### Option B — EfficientNetV2 (Medium or Large)
**What it is:** An improved and faster version of EfficientNet, trained with progressive learning (starts training on small images, increases size during training).

**Why it makes sense here:**
- Significantly higher accuracy than EfficientNetB2 with comparable compute
- Pre-trained on ImageNet-21k (21 000 classes) — much richer feature space
- Good balance between parameter count and accuracy

**Keras usage:** `keras.applications.EfficientNetV2M` or `keras.applications.EfficientNetV2L`

#### Option C — ConvNeXt
**What it is:** A "modernised" ResNet redesigned with lessons from Vision Transformers. Achieves ViT-level accuracy while remaining fully convolutional.

**Why it makes sense here:**
- Outperforms ResNet50 and EfficientNet on most benchmarks
- Fully convolutional → compatible with existing GradCAM visualisation
- Excellent with fine-grained visual recognition

**Keras usage:** Available via `keras_cv` or `timm`

### Advanced Techniques Worth Implementing

#### 1. Learning Rate Warm-up + Cosine Annealing Schedule
Instead of reducing LR by a fixed factor (`ReduceLROnPlateau`), use a cosine schedule:
- Start from a very low LR, warm up to the target LR over 5 epochs
- Then decay smoothly following a cosine curve

This is standard in transformer-based models and often improves final accuracy by 1–2%.

#### 2. Test-Time Augmentation (TTA)
During inference (prediction), apply random augmentations multiple times to the same image and average the predictions. No extra training required — pure inference-time boost.

Example: for one test image, generate 5 augmented versions → average 5 probability vectors → final prediction. Typically adds 0.5–2% accuracy.

#### 3. Layer-wise Learning Rate Decay
Apply different learning rates to different network layers:
- Layer 1–50 (early): LR × 0.01
- Layer 51–100 (mid): LR × 0.1
- Layer 101+ (late): LR × 1.0 (full rate)

This prevents destroying generic low-level features while still adapting high-level ones.

#### 4. Mixup or CutMix Augmentation
**Mixup**: blend two images together (e.g., 70% Monet + 30% Van Gogh) and train the model to predict a soft label (70% Monet, 30% Van Gogh). Forces the model to learn smoother decision boundaries.

**CutMix**: cut a rectangular patch from one image and paste it onto another, similarly blending labels. Tends to outperform Mixup on image classification.

These are especially useful with a small, imbalanced dataset.

#### 5. Class-Balanced Sampling
Instead of class-weighted loss, actively oversample rare classes (Dalí, Hopper) during training so they appear with equal frequency. This is stronger than reweighting alone.

#### 6. Stochastic Depth (Drop Path)
Randomly skip entire layers during training (analogous to dropout but at the layer level). Improves regularisation in deep networks. Already built into EfficientNetV2.

#### 7. Knowledge Distillation
Train a smaller "student" model to mimic the soft probability outputs of a larger "teacher" model (e.g., ResNet50). The student learns from the teacher's uncertainty — if the teacher says 60% Van Gogh / 30% Monet, the student learns the similarity between these styles.

---

## 14. Should We Use an Ensemble of More Than Two Models?

### The Case For (3+ Model Ensemble)

Ensembles work best when the individual models make **different errors** — their errors cancel out. The more architecturally diverse the models, the more complementary their error patterns:

- ResNet50 → residual connections, 50 layers, convolutional
- EfficientNetB2 → compound scaling, efficient architecture, convolutional
- ViT-B/16 → attention-based, global context, non-convolutional

A **3-model ensemble** (ResNet50 + EfficientNetV2 + ViT) would be highly diverse and likely outperform any 2-model combination.

### The Case Against (Diminishing Returns)

- Each additional model adds: training time, inference time, memory, and complexity
- The jump from 1 → 2 models is typically large (e.g., 78.8% → 80.5%)
- The jump from 2 → 3 is usually smaller (perhaps +0.5–1.0%)
- Beyond 3 models, gains become negligible

### Practical Recommendation

**Yes, a 3-model ensemble makes sense** for this project, especially if one of the new models is a ViT (architecturally very different from CNNs). A 4th model is unlikely to justify the compute cost unless you are optimising for a competition leaderboard.

**Optimal ensemble strategy:**
1. Train ResNet50 (or fine-tune existing) — strong baseline
2. Train EfficientNetV2M — strong CNN competitor
3. Train ViT-B/16 — attention-based, very different errors
4. Average their softmax outputs → ensemble prediction

Optionally: use **learned ensemble weights** (a simple logistic regression on their outputs) instead of a plain average, which can squeeze out another 0.2–0.5%.

---

## 15. Quick Reference Glossary

| Term | Definition |
|------|------------|
| **Tensor** | A multi-dimensional array (generalisation of matrix). Images are 3D tensors. |
| **Batch** | A small subset of training data (size 32 here) processed together before updating weights. |
| **Epoch** | One complete pass through the training set. |
| **Learning rate** | The step size for weight updates. Too large → unstable. Too small → slow. |
| **Loss function** | Measures how wrong the model's predictions are. We use cross-entropy. |
| **Backpropagation** | Algorithm for computing gradients of the loss w.r.t. every weight. |
| **Gradient descent** | Iteratively adjusting weights in the direction that reduces loss. |
| **Overfitting** | Model memorises training data; performs poorly on new data. |
| **Underfitting** | Model is too simple to capture the patterns in the data. |
| **Dropout** | Regularisation: randomly zero out neurons during training. |
| **BatchNorm** | Normalises layer outputs during training for stability. |
| **Softmax** | Converts raw scores to a probability distribution (sums to 1). |
| **Transfer learning** | Using a model pre-trained on one task as a starting point for another. |
| **Frozen layers** | Layers whose weights are not updated during training. |
| **Fine-tuning** | Slightly adjusting pre-trained weights for a new task (small LR). |
| **Data augmentation** | Creating new training examples by randomly transforming existing ones. |
| **Ensemble** | Combining predictions from multiple models, typically by averaging. |
| **GradCAM** | Technique that highlights which image regions most influenced a prediction. |
| **Label smoothing** | Prevents overconfidence by softening hard 0/1 targets to 0.1/0.9. |
| **Class weights** | Multipliers on the loss for rare classes, ensuring they influence training. |
| **Top-k accuracy** | Accuracy where the correct class is among the model's top k predictions. |
| **F1 score** | Harmonic mean of precision and recall; useful for imbalanced classes. |
| **ViT** | Vision Transformer; processes image patches with attention instead of convolutions. |
