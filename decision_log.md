# Decision Log — WikiArt Artist Classification Project

This document records every significant design decision made during the project,
including motivation, academic justification, implementation choices, and known tradeoffs.
Append new entries chronologically. Never delete existing entries.

---

## Phase 1 & 2 Decisions

Decisions 1–10 (from project inception through Phase 2 ensemble) are stored in
`outputs/decision_log.json` with full context, reasoning, action, and outcome fields.

---

## Phase 3 Decisions

---

### CutMix Augmentation — 2026-03-25

**Why we implemented this:**
The dataset is heavily class-imbalanced (Van Gogh: 1 322 images vs Salvador Dalí: 336).
Standard augmentation generates variations of individual images but does not address the
fundamental scarcity of rare-class samples. Additionally, several Impressionist artists
(Monet, Pissarro, Sisley) share highly similar visual styles — the model needs softer
decision boundaries between these artists rather than hard one-hot classification targets.
CutMix addresses both problems simultaneously: it implicitly oversamples rare classes
(by mixing them into training examples of majority classes) and trains soft mixed-label
targets that teach the model the degree of stylistic similarity between artists.

**Academic justification:**
Yun et al., "CutMix: Training Strategy that Makes Use of Sample Combinations",
ICCV 2019. https://arxiv.org/abs/1905.04899
CutMix was shown to outperform Mixup and Cutout on ImageNet and CIFAR classification
by forcing the model to identify discriminative regions rather than relying on
whole-image texture statistics. For art classification, where style is distributed
across the full canvas, CutMix encourages the model to recognise local style cues
(brushstroke texture, colour palette in partial views) rather than global composition.

**Key implementation choices:**
- Alpha = 1.0: Beta(1, 1) = Uniform[0, 1]. This gives equal probability to all cut sizes,
  matching the original paper's recommended setting for classification tasks.
- One-hot labels required: CutMix produces mixed soft targets (e.g., 0.6 × Monet + 0.4 × Pissarro).
  These cannot be expressed as integer labels. A separate `build_dataset_cutmix()` pipeline
  converts integer labels to one-hot and uses `CategoricalCrossentropy(label_smoothing=0.1)`.
- CutMix applied at batch level (after `.batch()`) using `tf.py_function`. This is necessary
  because CutMix requires shuffling pairs within a batch — it cannot be applied per-image.
- Base model: ResNet50 (same architecture as best Phase 2 model) to isolate the effect of CutMix.
- Strong augmentation applied before CutMix (flip, brightness, contrast, saturation, hue, crop).

**Known tradeoffs:**
- `tf.py_function` breaks static graph optimisation and reduces throughput vs pure TF ops.
- CutMix with crop augmentation may interact poorly (two composition-destroying transforms
  applied together). Monitor whether CutMix-trained ResNet50 outperforms the v2 ResNet50.
- One-hot labels are incompatible with `train_model()` (uses sparse loss internally).
  CutMix training uses `model.fit()` directly with manual compilation.

---

### Vision Transformer (ViT-B/16) — 2026-03-25

**Why we implemented this:**
All existing models (ResNet50, EfficientNetB2) are convolutional architectures.
CNNs apply localised filters that capture features in small spatial neighbourhoods.
Artistic style, however, often depends on whole-canvas composition: the relationship
between colour fields in Rothko, the spatial arrangement of figures in Raphael, or
the distribution of light in Rembrandt. These global, long-range dependencies are
poorly captured by CNNs but are directly modelled by the self-attention mechanism
in Vision Transformers. ViT is also the most architecturally different model available,
maximising the diversity of the ensemble's error patterns.

**Academic justification:**
Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition
at Scale", ICLR 2021. https://arxiv.org/abs/2010.11929
ViT-B/16 splits the image into 14×14 = 196 patches of 16×16 pixels. Each patch is
linearly projected to a 768-dimensional embedding. Twelve transformer encoder blocks
with multi-head self-attention then model pairwise relationships between ALL patches
simultaneously — unlike CNNs which only see local receptive fields. The CLS token
aggregates global information for classification. Pre-training on ImageNet-21k (21 000
categories, 14M images) provides richer feature representations than ImageNet-1k used
by ResNet/EfficientNet, giving ViT access to more diverse visual concepts.

**Key implementation choices:**
- TF Hub model: `sayakpaul/vit_b16_fe/1` (feature extraction variant, ImageNet-21k weights).
  This outputs a (batch, 768) CLS token embedding — compatible with our custom head.
- Input: [0, 1] float32 images. Our pipeline already normalises to [0, 1], so no additional
  preprocessing layer needed (unlike ResNet50 which requires `preprocess_input()`).
- Classification head: identical to ResNet50/EfficientNet — Dropout(0.5) → Dense(256) →
  Dropout(0.3) → Dense(23, softmax) — for fair comparison.
- Phase 2 fine-tuning: `hub.KerasLayer` does not expose indexed sub-layers, so we cannot
  do layer-index-based partial unfreezing like ResNet50 (layer 140). Instead we unfreeze
  the entire backbone with LR=1e-5, relying on the very small learning rate to prevent
  catastrophic forgetting. This is analogous to full fine-tuning at low LR.
- Same callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) as all other models.

**Known tradeoffs:**
- ViT requires `tensorflow_hub` as an additional dependency.
- `hub.KerasLayer` with `trainable=True` in Phase 2 updates ~86M parameters — significantly
  more than ResNet50 (~25M). Higher overfitting risk with our ~9 200 training images.
  EarlyStopping is critical here.
- ViT lacks convolutional layers, so Grad-CAM (which targets Conv2D activations) is not
  directly applicable. Attention rollout visualisation would be needed for interpretability.
- Training time is significantly longer than EfficientNetB2.

---

### Test-Time Augmentation (TTA) — 2026-03-25

**Why we implemented this:**
Model predictions are sensitive to minor input variation. Digitised paintings exhibit
natural variation: different museums scan at different resolutions and colour profiles,
lighting conditions vary, and JPEG compression introduces artefacts. A single deterministic
forward pass may land on the wrong side of a decision boundary for borderline examples.
TTA approximates Bayesian model averaging over the input distribution by generating N
randomly-augmented versions of each test image and averaging their softmax outputs.
This reduces prediction variance at zero additional training cost.

**Academic justification:**
Krizhevsky et al. first described test-time augmentation in AlexNet (NIPS 2012) as
"ten-crop" prediction. The theoretical justification from Bayesian deep learning
(Gal & Ghahramani, 2016) frames averaging over augmented inputs as Monte Carlo
integration over the input perturbation distribution, approximating the posterior
predictive distribution. For art classification, this is particularly motivated by the
heterogeneous digitisation conditions of paintings in the wild.

**Key implementation choices:**
- N = 5 augmented views: standard choice balancing accuracy improvement vs. inference cost.
  (N=1 is baseline; N > 10 shows diminishing returns per Lim et al. 2019.)
- Augmentation policy for TTA: ONLY horizontal flip + very small brightness/contrast (±5%).
  Explicitly NO crops, NO saturation, NO hue shifts.
  Rationale: The project established that random crops are destructive for paintings
  because composition IS a style signal (Decision #6). We must not apply at test time
  augmentations that the Phase 2 analysis showed to be harmful.
- Applied to each model independently before ensemble combination to measure per-model gain.

**Known tradeoffs:**
- 5× inference cost per model with TTA (5 passes per image instead of 1).
  For production use, 3 passes may be a better tradeoff.
- TTA with random seeds makes results non-deterministic across runs. Use `set_seeds()`
  before evaluation for reproducibility, though stochastic variation is expected.
- TTA improves accuracy on average but may hurt on easy, unambiguous examples where
  the augmentation introduces noise near a confident prediction.

---

### Per-Class Weighted Ensemble — 2026-03-25

**Why we implemented this:**
The Phase 2 ensemble (ResNet50 + EfficientNetB2) used simple equal-weight averaging.
This implicitly assumes every model is equally good at every class — which is false.
ResNet50 has high capacity for capturing complex style patterns but tends to overfit
rare artists (Dalí, Hopper). EfficientNetB2 has a smaller train-val gap (better
generalisation) but lower absolute accuracy. ViT's attention mechanism may excel at
globally-composed styles (Rothko, Klimt) where CNNs struggle. Per-class weighting
allows the ensemble to automatically defer to the strongest model for each artist,
without manual tuning.

**Academic justification:**
Wolpert, "Stacked Generalization", Neural Networks 1992. This is a simplified stacking
approach where the meta-learner is a fixed (non-trainable) per-class weight vector
rather than a learned secondary classifier. The weights are derived from per-class
validation accuracy — a principled, data-driven approach. This avoids the risk of
overfitting the meta-learner (which would require a held-out meta-validation set),
while still outperforming naive equal-weight averaging on imbalanced datasets where
class-specific model strengths differ significantly.

**Key implementation choices:**
- Weight source: per-class accuracy on the validation set (not test set, to prevent leakage).
- Normalisation: each column (class) of the weight matrix sums to 1.0, making the weighted
  sum a proper probability average for that class.
- Both equal-weight and per-class-weighted 3-model ensembles are evaluated, so the gain
  from weighting is directly measurable.
- The weight matrix is logged and printed as a table (see `print_weight_matrix()`) for
  interpretability — we can see which model the ensemble trusts most per artist.
- TTA inside the ensemble is evaluated as an optional upgrade if time permits.

**Known tradeoffs:**
- Weights derived from val accuracy ignore prediction calibration (a model with 80% class
  accuracy but poorly calibrated probabilities may not deserve a proportional weight).
  A calibration-aware alternative (e.g., temperature scaling + ECE-based weights) would
  be more principled.
- With only ~300 validation samples per class on average (and as few as 50 for Dalí),
  per-class accuracy estimates have high variance. Weights for rare classes are noisy.
- This approach does not allow the ensemble to learn cross-class correlations (e.g.,
  "when ResNet50 says Monet, it usually means Pissarro"). A full stacking approach
  (logistic regression on stacked predictions) would capture this but requires
  a separate meta-validation set.
