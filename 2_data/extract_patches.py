"""
Task 2 — Nucleus Patch Extraction
==================================
Extracts 100×100 nucleus-centred patches from the PUMA dataset ROIs.

Run once before any Task 2 training:

    python extract_patches.py

Output layout:
    Coumputer_Vision_Mini_Project_Data/task2_patches/
        train/
            Tumor/          ← 2500 .npy patches
            Lymphocyte/     ← 2500 .npy patches
            Histiocyte/     ← 2500 .npy patches
            metadata.csv
        validation/
            Tumor/          ← 700 .npy patches
            Lymphocyte/     ← 700 .npy patches
            Histiocyte/     ← 700 .npy patches
            metadata.csv
        contrastive/
            Tumor/          ← all nuclei from 20 held-out ROIs + spatially-safe extras
            Lymphocyte/
            Histiocyte/
            metadata.csv
        normalization_stats.json

Leakage prevention: 20 training ROIs are held out entirely for contrastive
pre-training (Part A). Unused nuclei from train ROIs that are ≥100px from
any training centroid are added as Part B. UUID overlap with training set
is always zero (verified before saving).

Test set (.npy files in Task2_Test_Set/) is already provided — not touched here.
"""

import json
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE      = Path(__file__).parent
DATA_ROOT  = _HERE.parent.parent / "Coumputer_Vision_Mini_Project_Data"
SPLITS_DIR = DATA_ROOT / "Dataset_Splits"
OUT_DIR    = DATA_ROOT / "task2_patches"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLASSES = {
    "nuclei_tumor":      "Tumor",
    "nuclei_lymphocyte": "Lymphocyte",
    "nuclei_histiocyte": "Histiocyte",
}

PATCH_SIZE          = 100    # must match test set format
TRAIN_N             = 2500   # patches per class for training
VAL_N               = 700    # patches per class for validation
CONTRASTIVE_N_ROIS  = 20     # full training ROIs held out for contrastive set
HISTO_META_FLOOR    = 0.22   # min metastatic share for Histiocyte sampling
SEED                = 42

# ---------------------------------------------------------------------------
# Optional dependencies (graceful fallbacks)
# ---------------------------------------------------------------------------

try:
    import tifffile as _tifffile
    _TIFFFILE = True
except ImportError:
    _TIFFFILE = False
    print("[WARN] tifffile not installed — falling back to PIL for TIF loading.")
    print("       uint16 TIFs may not load correctly.  pip install tifffile")

try:
    from shapely.geometry import shape as _shapely_shape
except ImportError:
    _shapely_shape = None
    print("[WARN] shapely not installed — using coordinate-averaging for centroids.")
    print("       pip install shapely")

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")


# ===========================================================================
# GeoJSON parsing
# ===========================================================================

def _load_geojson(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection":
            return data.get("features", [])
        return [data]
    raise ValueError(f"Unexpected GeoJSON structure: {path}")


def _get_classification(feature: dict) -> str:
    props = feature.get("properties", {})
    cls   = props.get("classification", {})
    if isinstance(cls, dict):
        return cls.get("name", "unknown")
    if isinstance(cls, str):
        return cls
    return props.get("name", "unknown")


def _compute_centroid(feature: dict):
    """Return (cx, cy) or None."""
    geom   = feature.get("geometry", {})
    gtype  = geom.get("type", "")
    coords = geom.get("coordinates", [])
    if not coords:
        return None

    if gtype == "Point":
        return (coords[0], coords[1])

    if gtype == "Polygon":
        if _shapely_shape is not None:
            try:
                c = _shapely_shape(geom).centroid
                return (c.x, c.y)
            except Exception:
                pass
        ext = coords[0]
        return (np.mean([p[0] for p in ext]), np.mean([p[1] for p in ext]))

    if gtype == "MultiPolygon":
        if _shapely_shape is not None:
            try:
                c = _shapely_shape(geom).centroid
                return (c.x, c.y)
            except Exception:
                pass
        xs, ys = [], []
        for poly in coords:
            ext = poly[0]
            xs.extend(p[0] for p in ext)
            ys.extend(p[1] for p in ext)
        return (np.mean(xs), np.mean(ys))

    return None


def _make_uuid(roi_name: str, centroid: tuple, cls_raw: str) -> str:
    key = f"{roi_name}_{centroid[0]:.2f}_{centroid[1]:.2f}_{cls_raw}"
    return hashlib.md5(key.encode()).hexdigest()


def _infer_sample_type(filename: str) -> str:
    fname = filename.lower()
    if "primary"    in fname:
        return "primary"
    if "metastatic" in fname:
        return "metastatic"
    return "unknown"


# ===========================================================================
# Image loading
# ===========================================================================

def _load_image_rgb(path: Path) -> np.ndarray:
    """Load as uint8 RGB (H, W, 3). Handles uint16 TIFs correctly."""
    if _TIFFFILE and str(path).lower().endswith((".tif", ".tiff")):
        arr = _tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = arr[..., :3]
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        elif arr.dtype != np.uint8:
            lo, hi = float(arr.min()), float(arr.max())
            arr = ((arr - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        return arr
    # PIL fallback
    img = np.array(Image.open(str(path)))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img.astype(np.uint8)


# ===========================================================================
# Nucleus catalogue
# ===========================================================================

def build_catalogue(split_dir: Path, split_name: str) -> pd.DataFrame:
    """Parse all ROIs and return a DataFrame of nucleus metadata (no patches yet)."""
    images_dir = split_dir / "image"
    if not images_dir.is_dir():
        images_dir = split_dir / "images"
    nuclei_dir = split_dir / "nuclei"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not nuclei_dir.is_dir():
        raise FileNotFoundError(f"Nuclei directory not found: {nuclei_dir}")

    tif_files = sorted(
        f for f in images_dir.iterdir()
        if f.suffix.lower() in (".tif", ".tiff")
    )
    print(f"\n{'='*60}")
    print(f"  Cataloguing nuclei: {split_name}  ({len(tif_files)} ROIs)")
    print(f"{'='*60}")

    records        = []
    class_counts   = defaultdict(int)
    ignored_counts = defaultdict(int)

    for tif_path in tqdm(tif_files, desc=f"  Parsing {split_name}"):
        roi_name = tif_path.stem

        # Find matching GeoJSON
        geojson_path = None
        for base in [roi_name, f"{roi_name}_nuclei"]:
            for ext in [".geojson", ".GeoJSON", ".json"]:
                candidate = nuclei_dir / (base + ext)
                if candidate.exists():
                    geojson_path = candidate
                    break
            if geojson_path:
                break
        if geojson_path is None:
            print(f"  [WARN] No GeoJSON for {roi_name}, skipping.")
            continue

        sample_type = _infer_sample_type(tif_path.name)
        features    = _load_geojson(geojson_path)

        for feat in features:
            cls_raw = _get_classification(feat)
            if cls_raw not in VALID_CLASSES:
                ignored_counts[cls_raw] += 1
                continue
            cls_label = VALID_CLASSES[cls_raw]
            centroid  = _compute_centroid(feat)
            if centroid is None:
                continue
            records.append({
                "roi_name":    roi_name,
                "image_path":  str(tif_path),
                "sample_type": sample_type,
                "class_raw":   cls_raw,
                "class_label": cls_label,
                "cx":          centroid[0],
                "cy":          centroid[1],
                "uuid":        _make_uuid(roi_name, centroid, cls_raw),
            })
            class_counts[cls_label] += 1

    df = pd.DataFrame(records)
    print(f"\n  Valid nuclei:")
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        print(f"    {cls:15s} {class_counts.get(cls, 0):>6d}")
    print(f"    {'TOTAL':15s} {len(df):>6d}")
    if ignored_counts:
        print(f"  Ignored types: {dict(ignored_counts)}")
    return df


# ===========================================================================
# Stratified sampling
# ===========================================================================

def _stratified_roi_sample(
    cls_df: pd.DataFrame, target: int, rng: np.random.RandomState
) -> pd.DataFrame:
    """Proportional sampling across ROIs with rounding fix."""
    if target <= 0 or len(cls_df) == 0:
        return cls_df.iloc[0:0]
    if len(cls_df) <= target:
        return cls_df

    roi_counts = cls_df.groupby("roi_name").size()
    total      = roi_counts.sum()
    allocation = (roi_counts / total * target).apply(np.floor).astype(int)

    # Distribute remainder to ROIs with largest fractional part
    remainder = target - allocation.sum()
    fractional = (roi_counts / total * target) - allocation
    if remainder > 0:
        top_rois = fractional.nlargest(int(remainder)).index
        allocation[top_rois] += 1

    roi_samples = []
    for roi_name, n_sample in allocation.items():
        if n_sample <= 0:
            continue
        roi_nuclei = cls_df[cls_df["roi_name"] == roi_name]
        n   = min(n_sample, len(roi_nuclei))
        idx = rng.choice(len(roi_nuclei), size=n, replace=False)
        roi_samples.append(roi_nuclei.iloc[idx])

    sampled = pd.concat(roi_samples, ignore_index=True)

    # Top up if slightly short due to rounding
    if len(sampled) < target:
        remaining = cls_df[~cls_df["uuid"].isin(sampled["uuid"])]
        extra = min(target - len(sampled), len(remaining))
        idx   = rng.choice(len(remaining), size=extra, replace=False)
        sampled = pd.concat([sampled, remaining.iloc[idx]], ignore_index=True)

    return sampled


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int,
    rois: set,
    seed: int = SEED,
) -> pd.DataFrame:
    """Sample n_per_class nuclei per class from the given ROIs."""
    rng    = np.random.RandomState(seed)
    roi_df = df[df["roi_name"].isin(rois)].copy()
    sampled_dfs = []

    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        cls_df    = roi_df[roi_df["class_label"] == cls]
        available = len(cls_df)
        target    = min(n_per_class, available)

        if available <= target:
            print(f"  {cls}: taking all {available} (target {n_per_class})")
            sampled_dfs.append(cls_df)
            continue

        # Histiocyte: guarantee metastatic floor
        if cls == "Histiocyte":
            meta_df    = cls_df[cls_df["sample_type"] == "metastatic"]
            primary_df = cls_df[cls_df["sample_type"] == "primary"]

            if len(meta_df) > 0 and len(primary_df) > 0:
                n_meta    = min(int(round(target * HISTO_META_FLOOR)), len(meta_df))
                meta_idx  = rng.choice(len(meta_df), size=n_meta, replace=False)
                meta_samp = meta_df.iloc[meta_idx]
                prim_samp = _stratified_roi_sample(primary_df, target - n_meta, rng)
                cls_samp  = pd.concat([meta_samp, prim_samp], ignore_index=True)
                if len(cls_samp) > target:
                    cls_samp = cls_samp.iloc[:target]
                print(f"  {cls}: {len(cls_samp)} "
                      f"(meta={len(meta_samp)}, primary={len(prim_samp)}) "
                      f"from {available}")
                sampled_dfs.append(cls_samp)
                continue

        cls_samp = _stratified_roi_sample(cls_df, target, rng)
        print(f"  {cls}: {len(cls_samp)} from {available} "
              f"across {cls_samp['roi_name'].nunique()} ROIs")
        sampled_dfs.append(cls_samp)

    return pd.concat(sampled_dfs, ignore_index=True)


# ===========================================================================
# Patch extraction & saving
# ===========================================================================

def _extract_patch(image: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """100×100 patch centred at (cx, cy), zero-padded at boundaries."""
    h, w = image.shape[:2]
    half  = PATCH_SIZE // 2

    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0, y0 = cx_i - half, cy_i - half
    x1, y1 = x0 + PATCH_SIZE, y0 + PATCH_SIZE

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)
    crop = image[iy0:iy1, ix0:ix1]

    patch = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    patch[iy0 - y0 : iy0 - y0 + crop.shape[0],
          ix0 - x0 : ix0 - x0 + crop.shape[1]] = crop
    return patch


def save_patches(df: pd.DataFrame, out_dir: Path, set_name: str) -> pd.DataFrame:
    """Extract and save patches; return df with patch_path column."""
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        cls_dir = out_dir / set_name / cls
        if cls_dir.exists():
            shutil.rmtree(cls_dir)
        cls_dir.mkdir(parents=True)

    # Sort by image path → only one image held in RAM at a time
    df = df.sort_values("image_path").reset_index(drop=True)

    current_img_path = None
    current_img      = None
    patch_paths      = []

    print(f"\n  Saving {set_name} patches ({len(df)} total)…")
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  Extracting {set_name}"):
        if row["image_path"] != current_img_path:
            current_img_path = row["image_path"]
            current_img      = _load_image_rgb(Path(current_img_path))

        patch    = _extract_patch(current_img, row["cx"], row["cy"])
        filename = f"{row['roi_name']}_{row['uuid'][:8]}.npy"
        out_path = out_dir / set_name / row["class_label"] / filename
        np.save(str(out_path), patch)
        patch_paths.append(str(out_path))

    df = df.copy()
    df["patch_path"] = patch_paths
    meta_path = out_dir / set_name / "metadata.csv"
    df.to_csv(str(meta_path), index=False)
    print(f"  Metadata → {meta_path}")
    return df


# ===========================================================================
# Verification
# ===========================================================================

def verify_patches(out_dir: Path, set_name: str) -> None:
    print(f"\n  Verifying {set_name}…")
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        cls_dir  = out_dir / set_name / cls
        npy_list = sorted(cls_dir.glob("*.npy"))
        if not npy_list:
            print(f"    {cls:15s}  [WARN] no files found")
            continue
        sample = np.load(str(npy_list[0]))
        ok     = sample.shape == (PATCH_SIZE, PATCH_SIZE, 3) and sample.dtype == np.uint8
        print(f"    {cls:15s}  n={len(npy_list):>5d}  "
              f"shape={sample.shape}  dtype={sample.dtype}  "
              f"[{'OK' if ok else 'ISSUE'}]")


# ===========================================================================
# Normalisation statistics
# ===========================================================================

def compute_normalization_stats(out_dir: Path) -> None:
    """Compute per-channel mean/std from all training patches, save JSON."""
    print("\n  Computing normalisation statistics from training patches…")
    per_class = {}
    for cls in ["Tumor", "Lymphocyte", "Histiocyte"]:
        cls_dir  = out_dir / "train" / cls
        npy_list = sorted(cls_dir.glob("*.npy"))
        if not npy_list:
            continue
        per_class[cls] = (
            np.stack([np.load(str(f)) for f in npy_list]).astype(np.float32) / 255.0
        )

    all_float = np.concatenate(list(per_class.values()), axis=0)
    n_total   = len(all_float)
    mean      = all_float.mean(axis=(0, 1, 2))
    std       = all_float.std(axis=(0, 1, 2))
    del all_float

    print(f"\n  Global stats ({n_total} patches):")
    print(f"    Mean : R={mean[0]:.4f}  G={mean[1]:.4f}  B={mean[2]:.4f}")
    print(f"    Std  : R={std[0]:.4f}  G={std[1]:.4f}  B={std[2]:.4f}")

    per_class_stats = {}
    for cls, arr in per_class.items():
        cm = arr.mean(axis=(0, 1, 2))
        cs = arr.std(axis=(0, 1, 2))
        print(f"    {cls:15s}  mean={cm.tolist()}  std={cs.tolist()}")
        per_class_stats[cls] = {
            "mean":  cm.tolist(),
            "std":   cs.tolist(),
            "count": len(arr),
        }

    stats = {
        "global_mean":   mean.tolist(),
        "global_std":    std.tolist(),
        "total_patches": n_total,
        "per_class":     per_class_stats,
    }
    stats_path = out_dir / "normalization_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved → {stats_path}")
    print(f"  Use in training:")
    print(f"    Normalize(mean={mean.tolist()}, std={std.tolist()})")


# ===========================================================================
# ROI-level train/contrastive split
# ===========================================================================

def split_rois_for_train_and_contrastive(
    df: pd.DataFrame,
    n_contrastive_rois: int = CONTRASTIVE_N_ROIS,
    seed: int = SEED,
) -> tuple:
    """
    Reserve n_contrastive_rois full training ROIs for contrastive pre-training.
    Stratified by primary/metastatic to keep both types in each group.

    Returns (train_rois: set, contrastive_rois: set).
    """
    rng       = np.random.RandomState(seed)
    all_rois  = df["roi_name"].unique()
    n_total   = len(all_rois)

    if n_contrastive_rois >= n_total:
        raise ValueError(
            f"n_contrastive_rois={n_contrastive_rois} >= total ROIs={n_total}"
        )

    print(f"\n  ROI-level split: {n_total} total train ROIs")
    print(f"  Reserving {n_contrastive_rois} ROIs for contrastive pre-training")

    roi_type        = df.groupby("roi_name")["sample_type"].first()
    primary_rois    = roi_type[roi_type == "primary"].index.tolist()
    metastatic_rois = roi_type[roi_type == "metastatic"].index.tolist()

    n_prim_cont = int(round(n_contrastive_rois * len(primary_rois) / n_total))
    n_prim_cont = min(n_prim_cont, len(primary_rois))
    n_meta_cont = n_contrastive_rois - n_prim_cont
    n_meta_cont = min(n_meta_cont, len(metastatic_rois))

    chosen = []
    if n_prim_cont > 0:
        idx = rng.choice(len(primary_rois), size=n_prim_cont, replace=False)
        chosen.extend([primary_rois[i] for i in idx])
    if n_meta_cont > 0:
        idx = rng.choice(len(metastatic_rois), size=n_meta_cont, replace=False)
        chosen.extend([metastatic_rois[i] for i in idx])
    # Top up if rounding left us short
    if len(chosen) < n_contrastive_rois:
        picked    = set(chosen)
        remaining = [r for r in all_rois if r not in picked]
        n_extra   = n_contrastive_rois - len(chosen)
        idx       = rng.choice(len(remaining), size=n_extra, replace=False)
        chosen.extend([remaining[i] for i in idx])

    contrastive_rois = set(chosen)
    train_rois       = set(all_rois) - contrastive_rois

    print(f"\n  Final ROI split:")
    print(f"    TRAIN ROIs       : {len(train_rois)}")
    print(f"    CONTRASTIVE ROIs : {len(contrastive_rois)}")

    for group_name, group_rois in [("TRAIN", train_rois), ("CONTRASTIVE", contrastive_rois)]:
        gdf = df[df["roi_name"].isin(group_rois)]
        print(f"\n    {group_name}:")
        print(f"      sample_types : {gdf['sample_type'].value_counts().to_dict()}")
        print(f"      class counts : {gdf['class_label'].value_counts().to_dict()}")

    return train_rois, contrastive_rois


def verify_no_leakage(train_df: pd.DataFrame, contrastive_df: pd.DataFrame) -> None:
    """Assert zero UUID overlap between train and contrastive sets."""
    train_uuids       = set(train_df["uuid"])
    contrastive_uuids = set(contrastive_df["uuid"])
    overlap           = train_uuids & contrastive_uuids
    if overlap:
        raise RuntimeError(f"UUID leakage! {len(overlap)} shared UUIDs.")
    print(f"\n  [OK] Zero UUID overlap — train: {len(train_df):,}  "
          f"contrastive: {len(contrastive_df):,}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print(f"\nData root  : {DATA_ROOT}")
    print(f"Output dir : {OUT_DIR}")

    # Skip if already extracted (check all three sets)
    train_tumor_dir       = OUT_DIR / "train"       / "Tumor"
    contrastive_tumor_dir = OUT_DIR / "contrastive" / "Tumor"
    if (train_tumor_dir.exists() and any(train_tumor_dir.glob("*.npy")) and
            contrastive_tumor_dir.exists() and any(contrastive_tumor_dir.glob("*.npy"))):
        print(
            f"\n[SKIP] Patches already exist at {OUT_DIR}\n"
            "       Delete the folder and re-run to re-extract."
        )
        sys.exit(0)

    if not SPLITS_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset_Splits not found at {SPLITS_DIR}\n"
            "Check that DATA_ROOT points to Coumputer_Vision_Mini_Project_Data."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Build nucleus catalogues
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 1: Build nucleus catalogues")
    print("=" * 60)
    train_all = build_catalogue(SPLITS_DIR / "train",      "train")
    val_all   = build_catalogue(SPLITS_DIR / "validation", "validation")

    # ------------------------------------------------------------------
    # Step 2: ROI-level train/contrastive split (leakage prevention)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2: ROI-level train/contrastive split")
    print("=" * 60)
    train_rois, contrastive_rois = split_rois_for_train_and_contrastive(
        train_all, n_contrastive_rois=CONTRASTIVE_N_ROIS, seed=SEED,
    )

    # ------------------------------------------------------------------
    # Step 3: Stratified sampling (train from train_rois only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3: Stratified sampling")
    print("=" * 60)

    print(f"\n  Training ({TRAIN_N}/class from {len(train_rois)} train ROIs):")
    train_sampled = stratified_sample(train_all, TRAIN_N, train_rois, seed=SEED)

    print(f"\n  Validation ({VAL_N}/class):")
    val_sampled = stratified_sample(
        val_all, VAL_N, set(val_all["roi_name"].unique()), seed=SEED
    )

    # ------------------------------------------------------------------
    # Step 4: Build contrastive set
    #   Part A — all nuclei from the 20 held-out contrastive ROIs
    #   Part B — spatially-safe unused nuclei from train ROIs
    #            (≥100px from any training centroid, cKDTree buffer)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 4: Build contrastive set (leakage-safe)")
    print("=" * 60)

    # Part A
    contrastive_df = train_all[
        train_all["roi_name"].isin(contrastive_rois)
    ].copy()
    print(f"\n  Part A — held-out contrastive ROIs: {len(contrastive_df):,} nuclei")

    # Part B — unused nuclei in train ROIs, spatially excluded from training set
    train_uuids = set(train_sampled["uuid"])
    candidates  = train_all[
        train_all["roi_name"].isin(train_rois)
        & ~train_all["uuid"].isin(train_uuids)
    ].copy()
    print(f"  Part B — candidate nuclei in train ROIs: {len(candidates):,}")

    buffer_px   = PATCH_SIZE   # 100px → zero pixel overlap guarantee
    train_by_roi = {
        roi: grp[["cx", "cy"]].values
        for roi, grp in train_sampled.groupby("roi_name")
    }
    safe_indices = []
    for roi_name, grp in candidates.groupby("roi_name"):
        if roi_name not in train_by_roi:
            # No training nuclei in this ROI — all candidates are safe
            safe_indices.extend(grp.index.tolist())
            continue
        tree  = cKDTree(train_by_roi[roi_name])
        dists, _ = tree.query(grp[["cx", "cy"]].values, k=1)
        safe_indices.extend(grp.index[dists >= buffer_px].tolist())

    expanded     = candidates.loc[safe_indices].copy()
    n_excluded   = len(candidates) - len(expanded)
    print(f"  Part B — excluded (within {buffer_px}px of training centroids): {n_excluded:,}")
    print(f"  Part B — spatially safe additions: {len(expanded):,}")

    contrastive_df = pd.concat([contrastive_df, expanded], ignore_index=True)
    print(f"\n  Contrastive total: {len(contrastive_df):,} patches "
          f"from {contrastive_df['roi_name'].nunique()} ROIs")

    verify_no_leakage(train_sampled, contrastive_df)

    # ------------------------------------------------------------------
    # Step 5: Extract and save patches
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 5: Extract and save patches")
    print("=" * 60)
    save_patches(train_sampled,  OUT_DIR, "train")
    save_patches(val_sampled,    OUT_DIR, "validation")
    save_patches(contrastive_df, OUT_DIR, "contrastive")

    # ------------------------------------------------------------------
    # Step 6: Verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 6: Verify")
    print("=" * 60)
    verify_patches(OUT_DIR, "train")
    verify_patches(OUT_DIR, "validation")
    verify_patches(OUT_DIR, "contrastive")

    # ------------------------------------------------------------------
    # Step 7: Normalisation statistics (from training set only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 7: Normalisation statistics")
    print("=" * 60)
    compute_normalization_stats(OUT_DIR)

    print("\n" + "=" * 60)
    print("  Done. Patches ready for Task 2a and 2b.")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
