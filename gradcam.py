"""
Grad-CAM: Gradient-weighted Class Activation Mapping.

Grad-CAM produces a heatmap highlighting which regions of the input
image were most important for the model's prediction. This is crucial
for interpretability — we can see whether the model focuses on
brushstrokes, composition, colour, or irrelevant artefacts.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization" (ICCV 2017).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import IMG_SIZE, CLASS_NAMES, PLOT_DIR
from utils import save_figure


def find_last_conv_layer(model):
    """
    Walk the model (including nested base models) to find
    the last convolutional layer — this is where Grad-CAM
    computes its activation maps.
    """
    # Check nested models (e.g. the ResNet/EfficientNet base)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            # This is a nested model — search inside it
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name, sub_layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    return None, None


def compute_gradcam(model, img_array, class_idx=None):
    """
    Compute the Grad-CAM heatmap for a single image.

    How it works:
      1. Forward-pass the image to get the prediction
      2. Compute gradients of the predicted class score w.r.t.
         the last convolutional layer's feature maps
      3. Global-average-pool the gradients to get per-channel weights
      4. Weighted-sum the feature maps → raw heatmap
      5. ReLU (keep only positive influence) + normalise to [0, 1]

    Parameters
    ----------
    model     : trained Keras model
    img_array : preprocessed image tensor, shape (1, H, W, 3)
    class_idx : class to explain (default: predicted class)

    Returns
    -------
    heatmap   : numpy array (H_feat, W_feat) in [0, 1]
    pred_idx  : predicted class index
    """
    base_name, conv_name = find_last_conv_layer(model)

    # Build a sub-model that outputs both the conv layer activations
    # and the final predictions
    if base_name:
        base_model = model.get_layer(base_name)
        conv_output = base_model.get_layer(conv_name).output
        # Build a model: input → [conv_activations, predictions]
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[base_model(model.layers[1](model.layers[0].output) if len(model.layers) > 2 else model.input),
                     model.output],
        )
        # Simpler approach: just use model input → conv output + predictions
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(base_name).get_layer(conv_name).output,
                     model.output],
        )
        # This won't work with nested models. Use a different approach:
        # Rebuild from the base model directly.
        pass

    # Universal approach: build gradient model by iterating through layers
    last_conv_output = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_output = sub_layer.output
                    break
        elif isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_output = layer.output
        if last_conv_output is not None:
            break

    # Fallback: use a simpler approach that works with any model structure
    return _compute_gradcam_universal(model, img_array, class_idx)


def _compute_gradcam_universal(model, img_array, class_idx=None):
    """
    Universal Grad-CAM that works with any model by finding
    the last Conv2D layer output through the model's computation graph.
    """
    # Find all Conv2D layers (including inside nested models)
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer)
        elif hasattr(layer, "layers"):
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    conv_layers.append(sub)

    if not conv_layers:
        raise ValueError("No Conv2D layers found in model.")

    last_conv = conv_layers[-1]

    # Build a model that outputs both conv activations and predictions
    # We need to trace through the model to get the conv output
    # Use GradientTape for a clean approach
    with tf.GradientTape() as tape:
        # Create an intermediate model that exposes the conv layer output
        # For nested models, we reconstruct the forward pass
        last_conv_output, predictions = _forward_with_conv_output(
            model, img_array, last_conv.name
        )
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_score = predictions[0, class_idx]

    # Gradients of the class score w.r.t. the conv layer output
    grads = tape.gradient(class_score, last_conv_output)

    # Global average pooling of gradients → importance weight per channel
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    heatmap = tf.reduce_sum(last_conv_output[0] * weights, axis=-1)

    # ReLU — we only care about features that positively influence the class
    heatmap = tf.maximum(heatmap, 0)

    # Normalise to [0, 1]
    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy(), int(class_idx)


def _forward_with_conv_output(model, img_array, target_conv_name):
    """
    Run a forward pass and capture both the target conv layer's
    output and the final prediction. Uses a temporary Keras Model.
    """
    # Find the target conv layer in the full model graph
    target_layer = None
    for layer in model.layers:
        if layer.name == target_conv_name:
            target_layer = layer
            break
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub.name == target_conv_name:
                    target_layer = sub
                    # For nested models, build a model from the nested base
                    base = layer
                    inner_model = tf.keras.Model(
                        inputs=base.input,
                        outputs=[sub.output, base.output],
                    )
                    # Run the full model manually
                    # Process input through layers before the base
                    x = img_array
                    for pre_layer in model.layers:
                        if pre_layer == base:
                            break
                        if pre_layer.__class__.__name__ == "InputLayer":
                            continue
                        x = pre_layer(x)
                    conv_out, base_out = inner_model(x)

                    # Continue through layers after the base
                    y = base_out
                    found_base = False
                    for post_layer in model.layers:
                        if post_layer == base:
                            found_base = True
                            continue
                        if not found_base:
                            continue
                        if post_layer.__class__.__name__ == "InputLayer":
                            continue
                        y = post_layer(y)

                    return conv_out, y

    # Simple (non-nested) model
    temp_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )
    conv_out, preds = temp_model(img_array)
    return conv_out, preds


def make_gradcam_overlay(img_path, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image.

    The heatmap is upscaled to match the image size and
    blended with the original using a jet colourmap.
    """
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)

    # Resize heatmap to image dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()

    # Apply jet colourmap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # drop alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Blend original image and heatmap
    overlay = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return img_array, overlay


# ──────────────────────────────────────────────
# High-level visualisation functions
# ──────────────────────────────────────────────
def visualise_gradcam_grid(model, test_paths, test_labels, model_name="model",
                            n_correct=4, n_wrong=4):
    """
    Show Grad-CAM heatmaps for a mix of correct and incorrect predictions.

    Why both: correct predictions show what the model learned to recognise.
    Incorrect predictions reveal what confused it — maybe it focuses on
    background instead of the actual painting style.
    """
    from data_loader import load_and_preprocess

    # Collect predictions for all provided images
    results = []
    for i, (path, true_label) in enumerate(zip(test_paths, test_labels)):
        img_tensor, _ = load_and_preprocess(path, true_label)
        img_batch = tf.expand_dims(img_tensor, 0)

        try:
            heatmap, pred_idx = _compute_gradcam_universal(model, img_batch)
            results.append({
                "path": path, "true": true_label, "pred": pred_idx,
                "correct": int(true_label) == pred_idx, "heatmap": heatmap,
            })
        except Exception:
            continue

        # Stop early once we have enough of each type
        correct_count = sum(1 for r in results if r["correct"])
        wrong_count = sum(1 for r in results if not r["correct"])
        if correct_count >= n_correct and wrong_count >= n_wrong:
            break

    # Separate correct and incorrect
    correct = [r for r in results if r["correct"]][:n_correct]
    wrong = [r for r in results if not r["correct"]][:n_wrong]
    selected = correct + wrong

    if not selected:
        print("Could not generate Grad-CAM for any images.")
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, item in enumerate(selected):
        original, overlay = make_gradcam_overlay(item["path"], item["heatmap"])
        true_name = CLASS_NAMES[item["true"]].replace("_", " ")
        pred_name = CLASS_NAMES[item["pred"]].replace("_", " ")
        color = "green" if item["correct"] else "red"
        label = f"True: {true_name}\nPred: {pred_name}"

        axes[i, 0].imshow(original)
        axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(item["heatmap"], cmap="jet")
        axes[i, 1].set_title("Grad-CAM heatmap", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(label, fontsize=10, color=color)
        axes[i, 2].axis("off")

    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{model_name}_gradcam.png")
    plt.show()
