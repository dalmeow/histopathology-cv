"""
Data exploration for tissue segmentation dataset.
Covers: image dimensions, class distribution, polygon stats, and sample visualizations.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from shapely.geometry import shape

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"

SPLITS = {
    "train":      DATA_ROOT / "train",
    "validation": DATA_ROOT / "validation",
    "test":       DATA_ROOT / "test",
}

# All tissue classes that appear in the geojson files
ALL_CLASSES = [
    "tissue_tumor",
    "tissue_stroma",
    "tissue_blood_vessel",
    "tissue_epidermis",
    "tissue_white_background",
    "tissue_necrosis",
]

# 3-class mapping required by the assignment
CLASS_MAP = {
    "tissue_tumor":             "Tumor",
    "tissue_stroma":            "Stroma",
    "tissue_blood_vessel":      "Other",
    "tissue_epidermis":         "Other",
    "tissue_white_background":  "Other",
    "tissue_necrosis":          "Other",
}

LABEL_COLORS = {
    "Tumor":  "#C80000",
    "Stroma": "#00C800",
    "Other":  "#0000C8",
}

OUTPUT_DIR = Path(__file__).parent / "exploration_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_samples(split_dir: Path):
    """Yield (image_path, tissue_geojson_path) pairs for a split."""
    img_dir    = split_dir / "image"
    tissue_dir = split_dir / "tissue"
    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem  # e.g. training_set_metastatic_roi_001
        geo_path = tissue_dir / f"{stem}_tissue.geojson"
        if geo_path.exists():
            yield img_path, geo_path


def load_geojson_classes(geo_path: Path) -> list[dict]:
    """Return list of {class_name, area_px2} for every feature in the file."""
    with open(geo_path) as f:
        gj = json.load(f)
    records = []
    for feat in gj.get("features", []):
        cls = feat["properties"]["classification"]["name"]
        geom = shape(feat["geometry"])
        records.append({"class": cls, "area": geom.area})
    return records


def image_stats(img_path: Path) -> dict:
    """Return basic stats for a TIFF image."""
    img = tifffile.imread(str(img_path))
    return {
        "height": img.shape[0],
        "width":  img.shape[1],
        "channels": img.shape[2] if img.ndim == 3 else 1,
        "dtype":  str(img.dtype),
        "mean":   img.mean(axis=(0, 1)).tolist() if img.ndim == 3 else float(img.mean()),
        "std":    img.std(axis=(0, 1)).tolist()  if img.ndim == 3 else float(img.std()),
    }


# ---------------------------------------------------------------------------
# 1. Scan entire dataset
# ---------------------------------------------------------------------------

def scan_dataset():
    print("\n" + "=" * 60)
    print("1. DATASET SCAN")
    print("=" * 60)

    all_heights, all_widths = [], []
    split_counts = {}
    raw_class_areas   = defaultdict(float)   # raw class name → total polygon area
    mapped_class_areas = defaultdict(float)  # Tumor/Stroma/Other → total polygon area
    raw_class_counts  = defaultdict(int)     # raw class → polygon count
    unseen_classes    = set()

    for split_name, split_dir in SPLITS.items():
        samples = list(iter_samples(split_dir))
        split_counts[split_name] = len(samples)
        print(f"\n  [{split_name}]  {len(samples)} samples")

        for img_path, geo_path in samples:
            stats = image_stats(img_path)
            all_heights.append(stats["height"])
            all_widths.append(stats["width"])

            for rec in load_geojson_classes(geo_path):
                cls = rec["class"]
                area = rec["area"]
                raw_class_areas[cls]  += area
                raw_class_counts[cls] += 1
                mapped = CLASS_MAP.get(cls)
                if mapped:
                    mapped_class_areas[mapped] += area
                else:
                    unseen_classes.add(cls)

    print(f"\n  Total samples : {sum(split_counts.values())}")
    print(f"  Split counts  : {split_counts}")

    print(f"\n  Image sizes (H×W):")
    print(f"    Heights  min={min(all_heights)}  max={max(all_heights)}  mean={np.mean(all_heights):.1f}")
    print(f"    Widths   min={min(all_widths)}   max={max(all_widths)}   mean={np.mean(all_widths):.1f}")
    unique_sizes = sorted(set(zip(all_heights, all_widths)))
    print(f"    Unique (H,W) combos: {unique_sizes}")

    print(f"\n  Raw tissue classes found:")
    total_area = sum(raw_class_areas.values())
    for cls in sorted(raw_class_areas):
        pct = 100 * raw_class_areas[cls] / total_area
        print(f"    {cls:<30s}  polygons={raw_class_counts[cls]:4d}  area={raw_class_areas[cls]:12.0f}  ({pct:.1f}%)")

    if unseen_classes:
        print(f"\n  WARNING – classes not in CLASS_MAP: {unseen_classes}")

    print(f"\n  3-class distribution (by polygon area):")
    total_mapped = sum(mapped_class_areas.values())
    for cls in ["Tumor", "Stroma", "Other"]:
        pct = 100 * mapped_class_areas[cls] / total_mapped
        print(f"    {cls:<10s}  {pct:.1f}%")

    return {
        "split_counts":         split_counts,
        "all_heights":          all_heights,
        "all_widths":           all_widths,
        "raw_class_areas":      dict(raw_class_areas),
        "raw_class_counts":     dict(raw_class_counts),
        "mapped_class_areas":   dict(mapped_class_areas),
    }


# ---------------------------------------------------------------------------
# 2. Per-image class coverage (train set)
# ---------------------------------------------------------------------------

def per_image_class_coverage():
    print("\n" + "=" * 60)
    print("2. PER-IMAGE CLASS COVERAGE (train set)")
    print("=" * 60)

    class_presence = defaultdict(int)  # how many images contain each 3-class label
    classes_per_image = []

    samples = list(iter_samples(SPLITS["train"]))
    for img_path, geo_path in samples:
        seen = set()
        for rec in load_geojson_classes(geo_path):
            mapped = CLASS_MAP.get(rec["class"])
            if mapped:
                seen.add(mapped)
        classes_per_image.append(len(seen))
        for c in seen:
            class_presence[c] += 1

    print(f"\n  Class presence across {len(samples)} training images:")
    for cls in ["Tumor", "Stroma", "Other"]:
        pct = 100 * class_presence[cls] / len(samples)
        print(f"    {cls:<10s}  {class_presence[cls]:3d} images  ({pct:.1f}%)")

    print(f"\n  Distinct classes per image:  "
          f"min={min(classes_per_image)}  max={max(classes_per_image)}  "
          f"mean={np.mean(classes_per_image):.2f}")

    return class_presence, classes_per_image


# ---------------------------------------------------------------------------
# 3. Visualize sample images with annotations
# ---------------------------------------------------------------------------

def visualize_samples(n_samples: int = 6):
    print("\n" + "=" * 60)
    print(f"3. SAMPLE VISUALIZATIONS  (n={n_samples})")
    print("=" * 60)

    samples = list(iter_samples(SPLITS["train"]))
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 4 * n_samples))
    fig.suptitle("Sample Images & Tissue Annotations", fontsize=14, y=1.01)

    for row, idx in enumerate(chosen):
        img_path, geo_path = samples[idx]
        img = tifffile.imread(str(img_path))

        # Build a color overlay from polygons
        overlay = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
        with open(geo_path) as f:
            gj = json.load(f)

        color_map_rgb = {
            "Tumor":  (200,   0,   0),
            "Stroma": (  0, 200,   0),
            "Other":  (  0,   0, 200),
        }

        # Draw Other first, then Stroma, then Tumor (priority order)
        priority = ["Other", "Stroma", "Tumor"]
        features_by_mapped = defaultdict(list)
        for feat in gj.get("features", []):
            raw_cls   = feat["properties"]["classification"]["name"]
            mapped    = CLASS_MAP.get(raw_cls, "Other")
            geom      = feat["geometry"]
            geom_type = geom["type"]
            if geom_type == "Polygon":
                rings = [geom["coordinates"][0]]
            elif geom_type == "MultiPolygon":
                rings = [poly[0] for poly in geom["coordinates"]]
            else:
                continue
            features_by_mapped[mapped].extend(rings)

        for mapped_cls in priority:
            rings = features_by_mapped.get(mapped_cls, [])
            if not rings:
                continue
            r, g, b = color_map_rgb[mapped_cls]
            for ring in rings:
                pts = np.array(ring, dtype=np.float32)[:, :2].astype(np.int32)
                cv2.fillPoly(overlay, [pts], color=(r, g, b))

        # Plot
        ax_img, ax_ann = axes[row]
        ax_img.imshow(img)
        ax_img.set_title(img_path.name, fontsize=8)
        ax_img.axis("off")

        ax_ann.imshow(img)
        ax_ann.imshow(overlay, alpha=0.45)
        ax_ann.set_title("Tissue annotation overlay", fontsize=8)
        ax_ann.axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color=hex_c, label=cls)
        for cls, hex_c in LABEL_COLORS.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    out_path = OUTPUT_DIR / "sample_annotations.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# 4. Summary plots
# ---------------------------------------------------------------------------

def plot_summary(stats: dict):
    print("\n" + "=" * 60)
    print("4. SUMMARY PLOTS")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Dataset Summary", fontsize=13)

    # (a) Split distribution
    ax = axes[0]
    splits = list(stats["split_counts"].keys())
    counts = [stats["split_counts"][s] for s in splits]
    bars = ax.bar(splits, counts, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.bar_label(bars, padding=3)
    ax.set_title("Samples per Split")
    ax.set_ylabel("Count")

    # (b) Raw class polygon area breakdown
    ax = axes[1]
    raw_areas = stats["raw_class_areas"]
    labels = [c.replace("tissue_", "").replace("_", "\n") for c in sorted(raw_areas)]
    values = [raw_areas[c] for c in sorted(raw_areas)]
    ax.barh(labels, values, color="#4C72B0")
    ax.set_title("Polygon Area by Raw Class (train+val+test)")
    ax.set_xlabel("Total area (px²)")
    ax.invert_yaxis()

    # (c) 3-class pie chart
    ax = axes[2]
    mapped = stats["mapped_class_areas"]
    cls_order = ["Tumor", "Stroma", "Other"]
    pie_vals  = [mapped.get(c, 0) for c in cls_order]
    pie_colors = [LABEL_COLORS[c] for c in cls_order]
    ax.pie(pie_vals, labels=cls_order, colors=pie_colors,
           autopct="%1.1f%%", startangle=140)
    ax.set_title("3-Class Area Distribution")

    out_path = OUTPUT_DIR / "dataset_summary.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_image_channels():
    """Assert all images across all splits are 3- or 4-channel (RGB/RGBA)."""
    print("\n" + "=" * 60)
    print("SANITY CHECK: Image channels")
    print("=" * 60)
    issues = []
    for split_name, split_dir in SPLITS.items():
        for img_path, _ in iter_samples(split_dir):
            img = tifffile.imread(str(img_path))
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] not in (3, 4)):
                issues.append(f"  {split_name}/{img_path.name}: shape={img.shape}")
    if issues:
        print("WARNING — unexpected channel counts:")
        for msg in issues:
            print(msg)
    else:
        print("  All images are RGB or RGBA.")


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def compute_mean_std(img_dir: Path) -> tuple[list[float], list[float]]:
    """
    Compute per-channel mean and std over all images in img_dir.
    Run once on your training image directory to get dataset-specific
    normalisation values.

    Example:
        mean, std = compute_mean_std(DATA_ROOT / "train" / "image")
        print(mean, std)
    """
    import tifffile as _tifffile
    pixel_sum   = np.zeros(3, dtype=np.float64)
    pixel_sq    = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for img_path in sorted(img_dir.glob("*.tif")):
        img = _tifffile.imread(str(img_path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3].astype(np.float64) / 255.0
        h, w = img.shape[:2]
        pixel_sum   += img.reshape(-1, 3).sum(axis=0)
        pixel_sq    += (img ** 2).reshape(-1, 3).sum(axis=0)
        pixel_count += h * w

    mean = pixel_sum / pixel_count
    std  = np.sqrt(pixel_sq / pixel_count - mean ** 2)
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_image_channels()
    stats = scan_dataset()
    per_image_class_coverage()
    plot_summary(stats)
    visualize_samples(n_samples=6)
    print("\nDone. All outputs saved to:", OUTPUT_DIR)
