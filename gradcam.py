"""
Grad-CAM: highlights which image regions drove the model's prediction.
Reference: Selvaraju et al., ICCV 2017.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from config import IMG_SIZE, CLASS_NAMES, PLOT_DIR
from utils import save_figure


def find_last_conv_layer(model):
    """Find the last Conv2D layer (searches nested models too)."""
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer.name, sub_layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return None, layer.name
    return None, None


def compute_gradcam(model, img_array, class_idx=None):
    """
    Compute Grad-CAM heatmap for a single image.
    Returns (heatmap, predicted_class_index).
    """
    base_name, conv_name = find_last_conv_layer(model)

    if base_name:
        base_model = model.get_layer(base_name)
        conv_output = base_model.get_layer(conv_name).output
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[base_model(model.layers[1](model.layers[0].output) if len(model.layers) > 2 else model.input),
                     model.output],
        )
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(base_name).get_layer(conv_name).output,
                     model.output],
        )
        pass

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

    return _compute_gradcam_universal(model, img_array, class_idx)


def _compute_gradcam_universal(model, img_array, class_idx=None):
    """Grad-CAM that works with any model structure (including nested bases)."""
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

    with tf.GradientTape() as tape:
        last_conv_output, predictions = _forward_with_conv_output(
            model, img_array, last_conv.name
        )
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_score = predictions[0, class_idx]

    grads = tape.gradient(class_score, last_conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(last_conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    if tf.reduce_max(heatmap) > 0:
        heatmap = heatmap / tf.reduce_max(heatmap)

    return heatmap.numpy(), int(class_idx)


def _forward_with_conv_output(model, img_array, target_conv_name):
    """Forward pass that captures both conv layer output and final prediction."""
    target_layer = None
    for layer in model.layers:
        if layer.name == target_conv_name:
            target_layer = layer
            break
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub.name == target_conv_name:
                    target_layer = sub
                    base = layer
                    inner_model = tf.keras.Model(
                        inputs=base.input,
                        outputs=[sub.output, base.output],
                    )
                    x = img_array
                    for pre_layer in model.layers:
                        if pre_layer == base:
                            break
                        if pre_layer.__class__.__name__ == "InputLayer":
                            continue
                        x = pre_layer(x)
                    conv_out, base_out = inner_model(x)

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

    temp_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )
    conv_out, preds = temp_model(img_array)
    return conv_out, preds


def make_gradcam_overlay(img_path, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on the original image."""
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img)

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()

    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    overlay = (img_array * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
    return img_array, overlay


def visualise_gradcam_grid(model, test_paths, test_labels, model_name="model",
                            n_correct=4, n_wrong=4):
    """Show Grad-CAM for a mix of correct and incorrect predictions."""
    from data_loader import load_and_preprocess

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

        correct_count = sum(1 for r in results if r["correct"])
        wrong_count = sum(1 for r in results if not r["correct"])
        if correct_count >= n_correct and wrong_count >= n_wrong:
            break

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

    fig.suptitle(f"Grad-CAM -- {model_name}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, f"{model_name}_gradcam.png")
    plt.show()
