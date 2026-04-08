"""
Exploratory Data Analysis (EDA) for the WikiArt dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from tqdm import tqdm

from config import DATA_DIR, CLASS_NAMES, PLOT_DIR
from utils import save_figure


# ──────────────────────────────────────────────
# 1. Catalogue every image path + label
# ──────────────────────────────────────────────
def build_file_list():
    """Walk dataset directory and return (file_paths, labels) lists."""
    file_paths, labels = [], []
    for artist in CLASS_NAMES:
        artist_dir = os.path.join(DATA_DIR, artist)
        for fname in os.listdir(artist_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(artist_dir, fname))
                labels.append(artist)
    return file_paths, labels


# ──────────────────────────────────────────────
# 2. Class distribution
# ──────────────────────────────────────────────
def plot_class_distribution(labels):
    """Bar chart of paintings per artist."""
    counts = Counter(labels)
    artists = [a for a, _ in counts.most_common()]
    values = [c for _, c in counts.most_common()]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("viridis", len(artists))
    ax.barh(artists, values, color=palette)
    ax.set_xlabel("Number of paintings")
    ax.set_title("Class distribution -- paintings per artist")
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)

    fig.tight_layout()
    save_figure(fig, "class_distribution.png")
    plt.show()

    return counts


# ──────────────────────────────────────────────
# 3. Image dimension statistics
# ──────────────────────────────────────────────
def analyse_image_dimensions(file_paths, sample_size=500):
    """Sample images and report width/height/aspect ratio stats."""
    rng = np.random.RandomState(42)
    indices = rng.choice(len(file_paths), size=min(sample_size, len(file_paths)), replace=False)

    widths, heights = [], []
    for idx in tqdm(indices, desc="Scanning image sizes"):
        with Image.open(file_paths[idx]) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths, heights = np.array(widths), np.array(heights)
    aspects = widths / heights

    print(f"Width  -- min: {widths.min()}, max: {widths.max()}, "
          f"mean: {widths.mean():.0f}, median: {np.median(widths):.0f}")
    print(f"Height -- min: {heights.min()}, max: {heights.max()}, "
          f"mean: {heights.mean():.0f}, median: {np.median(heights):.0f}")
    print(f"Aspect -- min: {aspects.min():.2f}, max: {aspects.max():.2f}, "
          f"mean: {aspects.mean():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(widths, heights, alpha=0.4, s=10)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].set_title("Image dimensions (sampled)")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(aspects, bins=40, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Aspect ratio (w / h)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Aspect ratio distribution")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "image_dimensions.png")
    plt.show()

    return widths, heights


# ──────────────────────────────────────────────
# 4. Sample grid
# ──────────────────────────────────────────────
def show_sample_images(file_paths, labels, n_per_artist=3):
    """Display a few example paintings for each artist."""
    n_artists = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_artists, n_per_artist,
                             figsize=(4 * n_per_artist, 3 * n_artists))

    artist_to_files = {}
    for fp, lbl in zip(file_paths, labels):
        artist_to_files.setdefault(lbl, []).append(fp)

    rng = np.random.RandomState(42)
    for row, artist in enumerate(CLASS_NAMES):
        files = artist_to_files[artist]
        chosen = rng.choice(files, size=n_per_artist, replace=False)
        for col, fp in enumerate(chosen):
            img = Image.open(fp).convert("RGB")
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(artist.replace("_", " "),
                                          fontsize=10, fontweight="bold")

    fig.suptitle("Sample paintings per artist", fontsize=14, y=1.01)
    fig.tight_layout()
    save_figure(fig, "sample_images.png")
    plt.show()


# ──────────────────────────────────────────────
# 5. Per-channel pixel statistics
# ──────────────────────────────────────────────
def compute_pixel_statistics(file_paths, sample_size=300):
    """Compute mean and std of pixel values per RGB channel."""
    rng = np.random.RandomState(42)
    indices = rng.choice(len(file_paths), size=min(sample_size, len(file_paths)), replace=False)

    channel_sum = np.zeros(3)
    channel_sq_sum = np.zeros(3)
    pixel_count = 0

    for idx in tqdm(indices, desc="Computing pixel stats"):
        img = Image.open(file_paths[idx]).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float64) / 255.0
        channel_sum += arr.sum(axis=(0, 1))
        channel_sq_sum += (arr ** 2).sum(axis=(0, 1))
        pixel_count += arr.shape[0] * arr.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)

    print(f"Per-channel mean (R, G, B): {mean}")
    print(f"Per-channel std  (R, G, B): {std}")
    print(f"ImageNet mean for reference: [0.485, 0.456, 0.406]")
    print(f"ImageNet std  for reference: [0.229, 0.224, 0.225]")

    return mean, std
