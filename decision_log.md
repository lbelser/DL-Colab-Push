# Decision Log -- WikiArt Artist Classification

Records design decisions with motivation and tradeoffs.
Append new entries at the bottom. Don't delete old ones.

---

## Phase 1 & 2 Decisions

Decisions 1-10 are in `outputs/decision_log.json` with full context/reasoning/outcome fields.

---

## Phase 3 Decisions

---

### CutMix Augmentation -- 2026-03-25

**Problem:** Class imbalance (Van Gogh 1,322 vs Dali 336) and similar Impressionists
(Monet, Pissarro, Sisley) that are hard to separate with hard one-hot labels.

**What we did:** CutMix (Yun et al., ICCV 2019) -- paste a random rectangular patch
from one training image onto another and mix their labels proportionally. So if 40%
of image A gets replaced by image B, the label becomes 0.6*A + 0.4*B.

**Why:** This does two things at once: rare classes get implicitly mixed into majority
class images (helps with imbalance), and the soft targets teach the model that
e.g. Monet and Pissarro are stylistically close instead of forcing a hard boundary.

**Implementation notes:**
- Alpha=1.0 for the Beta distribution (uniform cut sizes), same as the paper
- Has to run at batch level (after `.batch()`) because it needs pairs of images
- Uses `tf.py_function` which breaks graph optimisation but there's no clean alternative
- One-hot labels required -- compiled with CategoricalCrossentropy(label_smoothing=0.1)
- Using ResNet50 as base model to isolate CutMix effect from other changes

**Tradeoffs:**
- tf.py_function slows down the pipeline
- CutMix + random crop together might be too much (two transforms that mess with composition)
- Incompatible with train_model() since that uses sparse labels internally

---

### Vision Transformer (ViT-B/16) -- 2026-03-25

**Problem:** CNNs only see local neighborhoods through their filters. Painting style
often depends on the overall composition -- how colours are distributed across the
whole canvas, spatial arrangement of figures, etc.

**What we did:** Added a ViT-B/16 (Dosovitskiy et al., ICLR 2021) from TF Hub,
pretrained on ImageNet-21k. It splits the image into 14x14 = 196 patches of 16x16
pixels, projects each to 768 dimensions, then runs 12 transformer blocks with
self-attention over all patches at once.

**Why:** Self-attention can model relationships between any two patches regardless
of distance, so it should pick up on global composition patterns that CNNs miss.
Also, it's architecturally completely different from ResNet/EfficientNet, so its
errors will be different -- makes it a good third model for the ensemble.

**Implementation notes:**
- TF Hub model: `sayakpaul/vit_b16_fe/1` (outputs 768-dim CLS token)
- Expects [0,1] float inputs -- compatible with our pipeline as-is (no rescaling needed)
- Same classification head as ResNet/EfficientNet (Dropout->Dense(256)->Dropout->Dense(23))
- For fine-tuning: can't do partial unfreezing (hub.KerasLayer doesn't expose sub-layers),
  so we unfreeze everything and use LR=1e-5 to keep updates small
- 86M params when unfrozen -- much bigger than ResNet (25M). Overfitting risk is real
- Grad-CAM won't work on ViT (no Conv2D layers). Would need attention rollout for viz

**Tradeoffs:**
- Needs `tensorflow-hub` as extra dependency
- Slow to train (86M params to update in Phase 2)
- No easy interpretability (no Grad-CAM)

---

### Test-Time Augmentation (TTA) -- 2026-03-25

**Problem:** Predictions can change based on minor input differences (scan quality,
lighting, JPEG compression). A borderline image might flip to the wrong class.

**What we did:** At test time, generate 5 randomly augmented views of each image
and average their softmax predictions.

**Why:** Averages out the random variation from input perturbations. Like running
the same experiment 5 times and taking the mean -- reduces variance.

**Implementation notes:**
- Only gentle augmentation: horizontal flip + +-5% brightness/contrast
- NO random crops. Decision #6 showed crops hurt for paintings (composition = style signal)
- N=5 is standard. More than 10 gives diminishing returns
- 5x inference cost but zero training cost

**Tradeoffs:**
- 5x slower inference per model
- Slightly non-deterministic across runs (random augmentation)
- Can occasionally hurt on easy/unambiguous images where augmentation adds noise

---

### Per-Class Weighted Ensemble -- 2026-03-25

**Problem:** Phase 2 ensemble averaged all models equally, but different models
are better at different artists.

**What we did:** For each of the 23 artists, compute each model's accuracy on
the validation set. Use those per-class accuracies as weights (normalised to sum
to 1 per class). The ensemble then trusts whichever model is best for each artist.

**Why:** Equal averaging assumes all models are equally good at everything, which
isn't true. ResNet might be better at some artists, ViT at others. Per-class
weighting lets the ensemble pick the best model per artist automatically.

**Implementation notes:**
- Weights derived from val set (not test -- that would be leakage)
- Weight matrix shape: (n_models, 23). Each column sums to 1
- Both equal-weight and per-class-weighted 3-model ensembles evaluated for comparison
- Weight matrix is printed as a table so we can see which model dominates per artist

**Tradeoffs:**
- With ~50-100 val samples per class for some artists, accuracy estimates are noisy
- Ignores prediction calibration (a model with 80% acc but bad probabilities might
  not deserve a proportional weight)
- Doesn't capture cross-class patterns (e.g. "when ResNet says Monet it usually
  means Pissarro"). Full stacking would, but needs a meta-validation set
