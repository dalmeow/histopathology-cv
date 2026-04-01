"""
Task 2 — Nuclei Classification: Data Exploration
Covers: class inventory, primary/metastatic breakdown, per-image stats,
nucleus geometry & shape, patch overlap / data-leakage risk, per-class
colour profiles, tissue-context cross-reference, patch extraction
feasibility, Task2 test-set inspection, train-vs-test distribution
alignment, and sample visualisations.

Run with:
    /path/to/.env/bin/python data_exploration_task2.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from shapely.geometry import Point, shape

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_ROOT  = Path(__file__).parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Dataset_Splits"
TEST2_DIR  = Path(__file__).parent.parent / "Coumputer_Vision_Mini_Project_Data" / "Task2_Test_Set"
OUTPUT_DIR = Path(__file__).parent / "exploration_output_task2"
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = {
    "train":      DATA_ROOT / "train",
    "validation": DATA_ROOT / "validation",
    "test":       DATA_ROOT / "test",
}

VALID_CLASSES = ["nuclei_tumor", "nuclei_lymphocyte", "nuclei_histiocyte"]
SHORT = {"nuclei_tumor": "Tumor", "nuclei_lymphocyte": "Lymphocyte", "nuclei_histiocyte": "Histiocyte"}

CLASS_COLORS = {
    "Tumor":      "#C80000",
    "Lymphocyte": "#00C800",
    "Histiocyte": "#0000C8",
}

PATCH_SIZE = 100   # pixels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_type(stem: str) -> str:
    """Return 'primary' or 'metastatic' from a file stem."""
    return "primary" if "primary" in stem else "metastatic"


def iter_samples(split_dir: Path):
    """Yield (stem, img_path, nuclei_geo_path, tissue_geo_path) for a split."""
    img_dir     = split_dir / "image"
    nuclei_dir  = split_dir / "nuclei"
    tissue_dir  = split_dir / "tissue"
    for img_path in sorted(img_dir.glob("*.tif")):
        stem = img_path.stem
        n_path = nuclei_dir / f"{stem}_nuclei.geojson"
        t_path = tissue_dir / f"{stem}_tissue.geojson"
        if n_path.exists():
            yield stem, img_path, n_path, t_path if t_path.exists() else None


def polygon_props(geometry: dict) -> dict:
    """
    Compute area, perimeter, centroid, bounding box, and derived shape
    metrics from a GeoJSON geometry dict (Polygon or MultiPolygon).
    Uses shapely to handle all geometry types correctly.
    """
    geom = shape(geometry)
    # For MultiPolygon use the largest sub-polygon
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)

    area      = geom.area
    perimeter = geom.length
    cx        = geom.centroid.x
    cy        = geom.centroid.y
    x_min, y_min, x_max, y_max = geom.bounds
    bb_w = x_max - x_min
    bb_h = y_max - y_min

    equiv_diam   = math.sqrt(4 * area / math.pi) if area > 0 else 0.0
    circularity  = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    aspect_ratio = (bb_w / bb_h) if bb_h > 0 else 1.0

    return dict(
        area=area, perimeter=perimeter,
        cx=cx, cy=cy,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        bb_w=bb_w, bb_h=bb_h,
        equiv_diam=equiv_diam,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
    )


def extract_patch(img: np.ndarray, cx: float, cy: float, size: int = PATCH_SIZE) -> np.ndarray:
    """Extract a size×size patch centred at (cx, cy) with zero-padding if needed."""
    H, W = img.shape[:2]
    half = size // 2
    r0, r1 = int(round(cy)) - half, int(round(cy)) - half + size
    c0, c1 = int(round(cx)) - half, int(round(cx)) - half + size

    patch = np.zeros((size, size, 3), dtype=np.uint8)
    # source slice
    sr0, sr1 = max(r0, 0), min(r1, H)
    sc0, sc1 = max(c0, 0), min(c1, W)
    # destination slice
    dr0, dr1 = sr0 - r0, sr1 - r0
    dc0, dc1 = sc0 - c0, sc1 - c0

    if sr0 < sr1 and sc0 < sc1:
        patch[dr0:dr1, dc0:dc1] = img[sr0:sr1, sc0:sc1, :3]
    return patch


def load_tissue_shapes(tissue_path: Path):
    """Return list of (shapely_geom, tissue_class_name) for a tissue geojson."""
    if tissue_path is None or not tissue_path.exists():
        return []
    with open(tissue_path) as f:
        gj = json.load(f)
    result = []
    for feat in gj.get("features", []):
        try:
            geom = shape(feat["geometry"])
            cls  = feat["properties"]["classification"]["name"]
            result.append((geom, cls))
        except Exception:
            pass
    return result


def tissue_class_at(cx: float, cy: float, tissue_shapes) -> str:
    """Return tissue class containing point (cx, cy), or 'unknown'."""
    pt = Point(cx, cy)
    for geom, cls in tissue_shapes:
        if geom.contains(pt):
            return cls
    return "unknown"


# ---------------------------------------------------------------------------
# 1. Class inventory per split
# ---------------------------------------------------------------------------

def class_inventory():
    print("\n" + "=" * 60)
    print("1. CLASS INVENTORY PER SPLIT")
    print("=" * 60)

    rows = []
    for split_name, split_dir in SPLITS.items():
        valid_counts   = Counter()
        ignored_counts = Counter()
        for stem, _, n_path, _ in iter_samples(split_dir):
            with open(n_path) as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                cls = feat["properties"]["classification"]["name"]
                if cls in VALID_CLASSES:
                    valid_counts[cls] += 1
                else:
                    ignored_counts[cls] += 1

        total_valid   = sum(valid_counts.values())
        total_ignored = sum(ignored_counts.values())
        print(f"\n  [{split_name}]")
        for cls in VALID_CLASSES:
            print(f"    {SHORT[cls]:<12s}  {valid_counts[cls]:6d}")
        print(f"    {'TOTAL valid':<12s}  {total_valid:6d}")
        print(f"    ignored:  {dict(ignored_counts)}  (total {total_ignored})")

        # Feasibility check
        targets = {"train": 2500, "validation": 700}.get(split_name)
        if targets:
            print(f"\n  Extraction target: {targets} per class")
            for cls in VALID_CLASSES:
                ok = "OK" if valid_counts[cls] >= targets else "SHORTFALL"
                print(f"    {SHORT[cls]:<12s}  {valid_counts[cls]:6d} / {targets}  [{ok}]")

        rows.append((split_name, valid_counts, ignored_counts))
    return rows


# ---------------------------------------------------------------------------
# 2. Primary vs Metastatic breakdown
# ---------------------------------------------------------------------------

def primary_metastatic_breakdown():
    print("\n" + "=" * 60)
    print("2. PRIMARY vs METASTATIC BREAKDOWN")
    print("=" * 60)

    for split_name, split_dir in SPLITS.items():
        counts = defaultdict(lambda: Counter())  # sample_type -> {cls: count}
        n_roi  = Counter()
        for stem, _, n_path, _ in iter_samples(split_dir):
            stype = sample_type(stem)
            n_roi[stype] += 1
            with open(n_path) as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                cls = feat["properties"]["classification"]["name"]
                if cls in VALID_CLASSES:
                    counts[stype][cls] += 1

        print(f"\n  [{split_name}]  ROIs: {dict(n_roi)}")
        header = f"    {'type':<14s}" + "".join(f"  {SHORT[c]:<12s}" for c in VALID_CLASSES) + "  Total"
        print(header)
        for stype in ["metastatic", "primary"]:
            row = f"    {stype:<14s}"
            for cls in VALID_CLASSES:
                row += f"  {counts[stype][cls]:<12d}"
            row += f"  {sum(counts[stype].values())}"
            print(row)


# ---------------------------------------------------------------------------
# 3. Per-image nucleus stats
# ---------------------------------------------------------------------------

def per_image_stats():
    print("\n" + "=" * 60)
    print("3. PER-IMAGE NUCLEUS STATS (valid classes only)")
    print("=" * 60)

    for split_name, split_dir in SPLITS.items():
        per_img = []  # list of Counter per image
        for stem, _, n_path, _ in iter_samples(split_dir):
            cnt = Counter()
            with open(n_path) as f:
                gj = json.load(f)
            for feat in gj.get("features", []):
                cls = feat["properties"]["classification"]["name"]
                if cls in VALID_CLASSES:
                    cnt[cls] += 1
            per_img.append(cnt)

        totals = [sum(c.values()) for c in per_img]
        n_all3 = sum(1 for c in per_img if all(c[cl] > 0 for cl in VALID_CLASSES))
        print(f"\n  [{split_name}]  {len(per_img)} images")
        print(f"    Valid nuclei per image:  min={min(totals)}  max={max(totals)}  "
              f"mean={np.mean(totals):.1f}  median={np.median(totals):.0f}")
        print(f"    Images with all 3 classes present: {n_all3} / {len(per_img)}")
        for cls in VALID_CLASSES:
            vals = [c[cls] for c in per_img]
            print(f"    {SHORT[cls]:<12s}  min={min(vals)}  max={max(vals)}  "
                  f"mean={np.mean(vals):.1f}")


# ---------------------------------------------------------------------------
# 4. Nucleus geometry & shape
# ---------------------------------------------------------------------------

def geometry_stats():
    print("\n" + "=" * 60)
    print("4. NUCLEUS GEOMETRY & SHAPE (train set)")
    print("=" * 60)

    props_by_cls = defaultdict(list)   # SHORT name -> list of prop dicts

    for stem, _, n_path, _ in iter_samples(SPLITS["train"]):
        with open(n_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            cls = feat["properties"]["classification"]["name"]
            if cls not in VALID_CLASSES:
                continue
            p = polygon_props(feat["geometry"])
            props_by_cls[SHORT[cls]].append(p)

    metrics = ["area", "equiv_diam", "circularity", "aspect_ratio", "bb_w", "bb_h"]
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        props = props_by_cls[cls]
        print(f"\n  {cls}  (n={len(props)})")
        for m in metrics:
            vals = [p[m] for p in props]
            print(f"    {m:<14s}  mean={np.mean(vals):.2f}  "
                  f"std={np.std(vals):.2f}  "
                  f"p5={np.percentile(vals, 5):.2f}  "
                  f"p95={np.percentile(vals, 95):.2f}")

    # Plot: distributions of area, equiv_diam, circularity
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Nucleus Geometry Distributions (train)", fontsize=12)
    for ax, metric, xlabel in zip(axes,
                                  ["equiv_diam", "circularity", "aspect_ratio"],
                                  ["Equivalent Diameter (px)", "Circularity", "Aspect Ratio (W/H)"]):
        for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
            vals = [p[metric] for p in props_by_cls[cls]]
            ax.hist(vals, bins=60, alpha=0.55, label=cls, color=CLASS_COLORS[cls],
                    density=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUTPUT_DIR / "geometry_distributions.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"\n  Saved → {out}")

    return props_by_cls


# ---------------------------------------------------------------------------
# 5. Patch overlap / data-leakage risk
# ---------------------------------------------------------------------------

def patch_overlap_analysis():
    print("\n" + "=" * 60)
    print("5. PATCH OVERLAP / DATA-LEAKAGE RISK (train set)")
    print("=" * 60)

    overlap_counts = Counter()   # (cls_a, cls_b) sorted tuple -> overlap pair count
    total_pairs    = 0
    close_pairs    = 0           # centroid distance < PATCH_SIZE

    for stem, _, n_path, _ in iter_samples(SPLITS["train"]):
        with open(n_path) as f:
            gj = json.load(f)
        nuclei = []
        for feat in gj.get("features", []):
            cls = feat["properties"]["classification"]["name"]
            if cls not in VALID_CLASSES:
                continue
            p = polygon_props(feat["geometry"])
            nuclei.append((SHORT[cls], p["cx"], p["cy"]))

        # O(n²) distance check — acceptable for ~hundreds of nuclei per image
        for i in range(len(nuclei)):
            for j in range(i + 1, len(nuclei)):
                ca, xa, ya = nuclei[i]
                cb, xb, yb = nuclei[j]
                dist = math.sqrt((xa - xb)**2 + (ya - yb)**2)
                total_pairs += 1
                if dist < PATCH_SIZE:
                    close_pairs += 1
                    key = tuple(sorted([ca, cb]))
                    overlap_counts[key] += 1

    pct = 100 * close_pairs / total_pairs if total_pairs else 0
    print(f"\n  Total nucleus pairs (train): {total_pairs:,}")
    print(f"  Pairs with centroid distance < {PATCH_SIZE}px (overlap): "
          f"{close_pairs:,}  ({pct:.1f}%)")
    print(f"\n  Overlapping pairs by class combination:")
    for (ca, cb), cnt in sorted(overlap_counts.items(), key=lambda x: -x[1]):
        print(f"    {ca} × {cb:<12s}  {cnt:6d}")
    print(f"\n  NOTE: Contrastive set extraction must track used nucleus UUIDs")
    print(f"        to prevent leaking patches also present in the train set.")


# ---------------------------------------------------------------------------
# 6. Per-ROI class contribution (extraction planning)
# ---------------------------------------------------------------------------

def per_roi_contribution():
    print("\n" + "=" * 60)
    print("6. PER-ROI CLASS CONTRIBUTION (train set)")
    print("=" * 60)

    roi_data = []
    for stem, _, n_path, _ in iter_samples(SPLITS["train"]):
        with open(n_path) as f:
            gj = json.load(f)
        cnt = Counter()
        for feat in gj.get("features", []):
            cls = feat["properties"]["classification"]["name"]
            if cls in VALID_CLASSES:
                cnt[SHORT[cls]] += 1
        roi_data.append((stem, cnt))

    # Sort by histiocyte count (the rarest)
    roi_data.sort(key=lambda x: x[1]["Histiocyte"], reverse=True)

    # How many ROIs needed to hit target assuming we drain greedily
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        target  = 2500
        running = 0
        for i, (stem, cnt) in enumerate(roi_data):
            running += cnt[cls]
            if running >= target:
                print(f"  {cls:<12s}  reaches {target} after {i+1:3d} ROIs "
                      f"(cumulative: {running})")
                break
        else:
            print(f"  {cls:<12s}  NEVER reaches {target}  (total: {running})")

    # Top-10 ROIs for histiocyte (bottleneck class)
    print(f"\n  Top 10 ROIs by Histiocyte count:")
    for stem, cnt in roi_data[:10]:
        stype = sample_type(stem)
        print(f"    {stem:<55s}  H={cnt['Histiocyte']:4d}  "
              f"L={cnt['Lymphocyte']:4d}  T={cnt['Tumor']:4d}  [{stype}]")

    # Distribution plot: nuclei per ROI per class
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Per-ROI valid nucleus counts (train)", fontsize=12)
    for ax, cls in zip(axes, ["Tumor", "Lymphocyte", "Histiocyte"]):
        vals = sorted([cnt[cls] for _, cnt in roi_data], reverse=True)
        ax.bar(range(len(vals)), vals, color=CLASS_COLORS[cls], width=1.0)
        ax.set_title(cls)
        ax.set_xlabel("ROI rank")
        ax.set_ylabel("Count")
    plt.tight_layout()
    out = OUTPUT_DIR / "per_roi_contribution.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# 7. Per-class colour / intensity profiles
# ---------------------------------------------------------------------------

def colour_profiles(n_samples_per_cls: int = 300):
    print("\n" + "=" * 60)
    print("7. PER-CLASS COLOUR PROFILES (train set, sampled)")
    print("=" * 60)

    rng     = np.random.default_rng(42)
    patches_by_cls = defaultdict(list)
    budget  = {c: n_samples_per_cls for c in VALID_CLASSES}

    for stem, img_path, n_path, _ in iter_samples(SPLITS["train"]):
        if all(budget[c] <= 0 for c in VALID_CLASSES):
            break
        with open(n_path) as f:
            gj = json.load(f)
        feats = [ft for ft in gj.get("features", [])
                 if ft["properties"]["classification"]["name"] in VALID_CLASSES
                 and budget[ft["properties"]["classification"]["name"]] > 0]
        if not feats:
            continue
        img = tifffile.imread(str(img_path))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        rng.shuffle(feats)
        for feat in feats:
            cls = feat["properties"]["classification"]["name"]
            if budget[cls] <= 0:
                continue
            p   = polygon_props(feat["geometry"])
            patch = extract_patch(img, p["cx"], p["cy"])
            patches_by_cls[SHORT[cls]].append(patch)
            budget[cls] -= 1

    print(f"\n  Sampled patches: { {k: len(v) for k, v in patches_by_cls.items()} }")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Per-channel pixel intensity distribution by class (train)", fontsize=12)
    ch_names  = ["R", "G", "B"]
    ch_colors = ["red", "green", "blue"]

    for ax, cls in zip(axes, ["Tumor", "Lymphocyte", "Histiocyte"]):
        stack = np.stack(patches_by_cls[cls])   # (N, 100, 100, 3)
        pixels = stack.reshape(-1, 3).astype(np.float32)
        for ch, (ch_name, ch_col) in enumerate(zip(ch_names, ch_colors)):
            ax.hist(pixels[:, ch], bins=64, range=(0, 255),
                    alpha=0.55, label=ch_name, color=ch_col, density=True)
        means = pixels.mean(axis=0)
        stds  = pixels.std(axis=0)
        ax.set_title(f"{cls}\nmean RGB=({means[0]:.0f},{means[1]:.0f},{means[2]:.0f})  "
                     f"std=({stds[0]:.0f},{stds[1]:.0f},{stds[2]:.0f})", fontsize=8)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "colour_profiles.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved → {out}")

    # Print mean/std per class per channel
    print(f"\n  Per-class mean ± std (R / G / B):")
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        stack  = np.stack(patches_by_cls[cls]).reshape(-1, 3).astype(np.float32)
        means  = stack.mean(axis=0) / 255.0
        stds   = stack.std(axis=0)  / 255.0
        print(f"    {cls:<12s}  R={means[0]:.3f}±{stds[0]:.3f}  "
              f"G={means[1]:.3f}±{stds[1]:.3f}  B={means[2]:.3f}±{stds[2]:.3f}")

    return patches_by_cls


# ---------------------------------------------------------------------------
# 8. Tissue-context cross-reference
# ---------------------------------------------------------------------------

def tissue_context():
    print("\n" + "=" * 60)
    print("8. TISSUE CONTEXT CROSS-REFERENCE (train set, sampled)")
    print("=" * 60)

    # nucleus_class → tissue_class → count
    cross = defaultdict(Counter)
    n_processed = 0

    for stem, _, n_path, t_path in iter_samples(SPLITS["train"]):
        tissue_shapes = load_tissue_shapes(t_path)
        if not tissue_shapes:
            continue
        with open(n_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            cls = feat["properties"]["classification"]["name"]
            if cls not in VALID_CLASSES:
                continue
            p = polygon_props(feat["geometry"])
            t_cls = tissue_class_at(p["cx"], p["cy"], tissue_shapes)
            cross[SHORT[cls]][t_cls] += 1
        n_processed += 1

    # Collect all tissue classes seen
    all_t = sorted({tc for counts in cross.values() for tc in counts})
    short_t = {tc: tc.replace("tissue_", "") for tc in all_t}

    header = f"  {'nucleus \\ tissue':<14s}" + "".join(f"  {short_t[t]:<22s}" for t in all_t)
    print(f"\n{header}")
    for n_cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        total = sum(cross[n_cls].values())
        row   = f"  {n_cls:<14s}"
        for t in all_t:
            cnt = cross[n_cls][t]
            pct = 100 * cnt / total if total else 0
            row += f"  {cnt:6d} ({pct:4.1f}%)        "
        print(row)

    # Heatmap
    mat_labels_n = ["Tumor", "Lymphocyte", "Histiocyte"]
    mat_labels_t = [short_t[t] for t in all_t]
    mat = np.array([[cross[n][t] for t in all_t] for n in mat_labels_n], dtype=float)
    mat_pct = mat / mat.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(max(8, len(all_t) * 1.5), 4))
    im = ax.imshow(mat_pct, cmap="Blues")
    ax.set_xticks(range(len(mat_labels_t)))
    ax.set_xticklabels(mat_labels_t, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat_labels_n)))
    ax.set_yticklabels(mat_labels_n, fontsize=9)
    ax.set_title("Nucleus class × Tissue context  (% row-normalised)", fontsize=10)
    plt.colorbar(im, ax=ax, label="%")
    for i in range(len(mat_labels_n)):
        for j in range(len(mat_labels_t)):
            ax.text(j, i, f"{mat_pct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=7, color="black" if mat_pct[i, j] < 60 else "white")
    plt.tight_layout()
    out = OUTPUT_DIR / "tissue_context_heatmap.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# 9. Patch extraction feasibility (zero-padding stats)
# ---------------------------------------------------------------------------

def patch_feasibility():
    print("\n" + "=" * 60)
    print("9. PATCH EXTRACTION FEASIBILITY (train set)")
    print("=" * 60)

    half = PATCH_SIZE // 2
    stats = defaultdict(lambda: {"total": 0, "needs_pad": 0})

    for stem, img_path, n_path, _ in iter_samples(SPLITS["train"]):
        img = tifffile.imread(str(img_path))
        H, W = img.shape[:2]
        with open(n_path) as f:
            gj = json.load(f)
        for feat in gj.get("features", []):
            cls = feat["properties"]["classification"]["name"]
            if cls not in VALID_CLASSES:
                continue
            p   = polygon_props(feat["geometry"])
            cx, cy = p["cx"], p["cy"]
            needs_pad = (cx - half < 0 or cx + half > W or
                         cy - half < 0 or cy + half > H)
            stats[SHORT[cls]]["total"]     += 1
            stats[SHORT[cls]]["needs_pad"] += int(needs_pad)

    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        t = stats[cls]["total"]
        p = stats[cls]["needs_pad"]
        print(f"  {cls:<12s}  total={t:6d}  needs_padding={p:5d}  "
              f"({100*p/t:.1f}%)")


# ---------------------------------------------------------------------------
# 10. Task2 test-set inspection
# ---------------------------------------------------------------------------

def inspect_test2():
    print("\n" + "=" * 60)
    print("10. TASK2 TEST-SET INSPECTION")
    print("=" * 60)

    npy_files = sorted(TEST2_DIR.glob("*.npy"))
    print(f"\n  Total .npy files: {len(npy_files)}")

    # Parse filenames
    cls_counts    = Counter()
    stype_counts  = Counter()
    cls_stype     = defaultdict(Counter)
    roi_counts    = Counter()

    pattern = re.compile(
        r"^(?P<stype>test_set_(?:metastatic|primary))_roi_(?P<roi>\d+)"
        r"_nuclei_(?P<cls>\w+)_"
    )

    shapes_seen = set()
    dtypes_seen = set()
    sample_patches = defaultdict(list)

    for fp in npy_files:
        m = pattern.match(fp.name)
        if not m:
            continue
        stype = "primary" if "primary" in m.group("stype") else "metastatic"
        cls   = "nuclei_" + m.group("cls")
        short = SHORT.get(cls, m.group("cls"))
        roi   = m.group("roi")

        cls_counts[short]         += 1
        stype_counts[stype]       += 1
        cls_stype[short][stype]   += 1
        roi_counts[roi]           += 1

        if len(sample_patches[short]) < 8:
            arr = np.load(str(fp))
            shapes_seen.add(arr.shape)
            dtypes_seen.add(str(arr.dtype))
            sample_patches[short].append(arr)

    print(f"\n  Shapes seen: {shapes_seen}")
    print(f"  dtypes seen: {dtypes_seen}")
    print(f"\n  Class counts:")
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        print(f"    {cls:<12s}  total={cls_counts[cls]:4d}  "
              f"metastatic={cls_stype[cls]['metastatic']:4d}  "
              f"primary={cls_stype[cls]['primary']:4d}")
    print(f"\n  Sample type totals:  {dict(stype_counts)}")
    print(f"  Unique ROIs:         {len(roi_counts)}")

    # Sample grid
    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    fig.suptitle("Task2 Test Set — sample patches per class", fontsize=11)
    for row, cls in enumerate(["Tumor", "Lymphocyte", "Histiocyte"]):
        for col in range(8):
            ax = axes[row][col]
            if col < len(sample_patches[cls]):
                ax.imshow(sample_patches[cls][col])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=90, labelpad=4)
    plt.tight_layout()
    out = OUTPUT_DIR / "test2_sample_patches.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"\n  Saved → {out}")

    return sample_patches


# ---------------------------------------------------------------------------
# 11. Train vs Test2 distribution alignment
# ---------------------------------------------------------------------------

def train_vs_test_alignment(patches_by_cls: dict, test_patches: dict):
    print("\n" + "=" * 60)
    print("11. TRAIN vs TEST2 DISTRIBUTION ALIGNMENT")
    print("=" * 60)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle("Train vs Test2 patch intensity per class × channel", fontsize=11)
    ch_names  = ["R", "G", "B"]
    ch_colors = ["red", "green", "blue"]

    for row, cls in enumerate(["Tumor", "Lymphocyte", "Histiocyte"]):
        train_px = np.stack(patches_by_cls[cls]).reshape(-1, 3).astype(np.float32)
        test_arr = test_patches.get(cls, [])
        if not test_arr:
            continue
        test_px  = np.stack(test_arr).reshape(-1, 3).astype(np.float32)

        for col, (ch, ch_col) in enumerate(zip(range(3), ch_colors)):
            ax = axes[row][col]
            ax.hist(train_px[:, ch], bins=64, range=(0, 255),
                    alpha=0.55, label="train", color=ch_col, density=True)
            ax.hist(test_px[:, ch],  bins=64, range=(0, 255),
                    alpha=0.55, label="test2", color="grey", density=True)
            ax.set_title(f"{cls} / {ch_names[ch]}", fontsize=8)
            ax.set_xlabel("Intensity")
            if col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    out = OUTPUT_DIR / "train_vs_test_alignment.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"\n  Saved → {out}")

    # Numeric summary: mean absolute difference per channel
    print(f"\n  Mean absolute difference in channel means (train vs test2):")
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        if cls not in patches_by_cls or cls not in test_patches:
            continue
        train_px = np.stack(patches_by_cls[cls]).reshape(-1, 3).astype(np.float32)
        test_px  = np.stack(test_patches[cls]).reshape(-1, 3).astype(np.float32)
        diff = np.abs(train_px.mean(axis=0) - test_px.mean(axis=0))
        print(f"    {cls:<12s}  ΔR={diff[0]:.1f}  ΔG={diff[1]:.1f}  ΔB={diff[2]:.1f}  "
              f"(out of 255)")


# ---------------------------------------------------------------------------
# 12. Visual sample patches
# ---------------------------------------------------------------------------

def visualise_samples(patches_by_cls: dict):
    print("\n" + "=" * 60)
    print("12. VISUAL SAMPLE PATCHES (train set)")
    print("=" * 60)

    n_cols = 10
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 1.5, 6))
    fig.suptitle("Sample 100×100 patches — train (Tumor / Lymphocyte / Histiocyte)", fontsize=11)

    rng = np.random.default_rng(0)
    for row, cls in enumerate(["Tumor", "Lymphocyte", "Histiocyte"]):
        pool = patches_by_cls[cls]
        idxs = rng.choice(len(pool), size=min(n_cols, len(pool)), replace=False)
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(idxs):
                ax.imshow(pool[idxs[col]])
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=9, rotation=90, labelpad=4)

    plt.tight_layout()
    out = OUTPUT_DIR / "sample_patches_train.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    class_inventory()
    primary_metastatic_breakdown()
    per_image_stats()
    props_by_cls = geometry_stats()
    patch_overlap_analysis()
    per_roi_contribution()
    patches_by_cls = colour_profiles(n_samples_per_cls=500)
    tissue_context()
    patch_feasibility()
    test_patches = inspect_test2()
    train_vs_test_alignment(patches_by_cls, test_patches)
    visualise_samples(patches_by_cls)

    print(f"\nDone. All outputs saved to: {OUTPUT_DIR}")
