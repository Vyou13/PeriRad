import os
import cv2
import numpy as np
import pandas as pd

# =========================================================================
# ROSENHOLTZ SUBBAND ENTROPY CLUTTER MEASURE (Rosenholtz et al., 2007)
# =========================================================================
# This is the *true* clutter measure, not saliency. It quantifies the number
# of bits required to encode the image at a given fidelity — i.e. how much
# "stuff" is present. The algorithm:
#
#   1. Convert RGB -> CIELab.
#   2. Decompose L, a, b into subbands with a steerable pyramid.
#   3. For each subband, bin the coefficients (nbins = sqrt(#coeffs)) and
#      compute Shannon entropy  H = -sum_i p_i log(p_i).
#   4. Sum subband entropies separately for L and for (a, b).
#   5. Weighted sum: 0.84 * L + 0.08 * a + 0.08 * b.
#
# Requires pyrtools (the canonical steerable-pyramid implementation):
#     pip install pyrtools
# =========================================================================

import pyrtools as pt

# Pyramid configuration
PYR_HEIGHT = 3       # number of scales
PYR_ORDER = 3        # order -> (order+1) = 4 orientations per scale

# Channel weights (from the paper)
W_LUM = 0.84
W_CHROM = 0.08


def subband_entropy_channel(channel):
    """Sum of Shannon entropies over all steerable-pyramid subbands of a 2D channel."""
    channel = channel.astype(np.float64)

    # Steerable pyramid: returns oriented bandpass subbands + hi/lo residuals.
    pyr = pt.pyramids.SteerablePyramidSpace(channel, height=PYR_HEIGHT, order=PYR_ORDER)

    total_entropy = 0.0
    for key, band in pyr.pyr_coeffs.items():
        # Skip the lowpass residual (it carries no oriented detail / not a true subband).
        if key == 'residual_lowpass':
            continue
        total_entropy += _shannon_entropy(band.ravel())
    return total_entropy


def _shannon_entropy(coeffs):
    """Shannon entropy of coefficients, binned into sqrt(N) bins."""
    n = coeffs.size
    if n == 0:
        return 0.0
    nbins = max(1, int(np.sqrt(n)))
    if nbins == 1:
        return 0.0

    hist, _ = np.histogram(coeffs, bins=nbins)
    p = hist.astype(np.float64)
    total = p.sum()
    if total == 0:
        return 0.0
    p = p / total
    p = p[p > 0]  # 0 * log(0) := 0
    return float(-np.sum(p * np.log(p)))


def subband_entropy_clutter(bgr_img):
    """Weighted L/a/b subband-entropy clutter for a BGR image."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2Lab).astype(np.float64)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    e_L = subband_entropy_channel(L)
    e_a = subband_entropy_channel(a)
    e_b = subband_entropy_channel(b)

    return W_LUM * e_L + W_CHROM * e_a + W_CHROM * e_b


# -------------------------------------------------------------------------
# CONFIGURATION (matched to Rosenholtz_saliency.py)
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"

CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "clinical_data_clean.csv"),
    os.path.join(BASE_DIR, "clinical_data_clean.csv"),
]
CSV_PATH = next((path for path in CSV_CANDIDATES if os.path.exists(path)), CSV_CANDIDATES[0])

RING_DISTANCE_PX = 86  # 10mm-57 expansion footprint, 15mm- 86
MM_TO_PX = 5.7         # Conversion factor: 1mm = 5.7 pixels (based on 57px = 10mm)

FILTER_BY_SIZE = True      # toggle on/off
MIN_RADIUS_MM = 15.0        # lower bound (inclusive); set to None to ignore
MAX_RADIUS_MM = 15.0       # upper bound (inclusive); set to None to ignore

# Output filename reflects the active size filter so different ranges don't overwrite each other.
size_tag = f"_{MIN_RADIUS_MM:g}-{MAX_RADIUS_MM:g}mm" if FILTER_BY_SIZE else ""
OUTPUT_CSV = os.path.join(BASE_DIR, "results", f"subbandEntropy15mm{size_tag}.csv")

#Images to exclude from the analysis (matched against the Image_ID column).
# Add IDs here to drop them, e.g. EXCLUDE_IMAGES = ["JPCLN084.png", "JPCLN106.png"].
# Matching ignores the file extension AND any spaces, so "JPCLN 084",
# "JPCLN084", and "JPCLN084.png" all match the same image.
EXCLUDE_IMAGES = [
    'JPCLN119', 'JPCLN133', 'JPCLN138', 'JPCLN142', 'JPCLN143',
    'JPCLN144', 'JPCLN145', 'JPCLN146', 'JPCLN153', 'JPCLN154',
]

EXCLUDE_IMAGE_STEMS = {
    os.path.splitext(str(image_id))[0].replace(" ", "").upper()
    for image_id in EXCLUDE_IMAGES
}


def is_excluded_image(image_id):
    normalized = os.path.splitext(str(image_id))[0].replace(" ", "").upper()
    return normalized in EXCLUDE_IMAGE_STEMS


def circular_mask(shape, cx, cy, radius):
    """Boolean mask: True inside a circle of `radius` centered at (cx, cy)."""
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def crop_to_region(img, region_mask):
    """Bounding-box crop of img covering region_mask, with all pixels OUTSIDE the
    region filled with the region's mean (per channel).

    Steerable pyramids need a full rectangular array, so a non-rectangular region
    (e.g. the perinodular ring, which excludes the nodule core) can't be fed in
    directly. Filling the excluded pixels with the region mean keeps the array
    rectangular while ensuring those pixels contribute (near) zero clutter of
    their own -- so only the region's actual texture drives the entropy."""
    ys, xs = np.where(region_mask)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop = img[y0:y1, x0:x1].copy()
    sub_mask = region_mask[y0:y1, x0:x1]

    if crop.ndim == 3:
        for c in range(crop.shape[2]):
            ch = crop[:, :, c]
            ch[~sub_mask] = ch[sub_mask].mean()
    else:
        crop[~sub_mask] = crop[sub_mask].mean()
    return crop


# =========================================================================
# 1. READ THE CLEAN CSV FILE 
# =========================================================================
print("Reading clean clinical data CSV...")
clinical_df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(clinical_df)} nodule records")

coordinate_lookup = {}
for _, row in clinical_df.iterrows():
    img_id = row['Image_ID'].replace('.IMG', '.png')
    if is_excluded_image(img_id):
        print(f"Skipping excluded image: {img_id}")
        continue

    cx = int(row['X_Coord']) if pd.notna(row['X_Coord']) else None
    cy = int(row['Y_Coord']) if pd.notna(row['Y_Coord']) else None
    radius_mm = row['Radius_mm'] if pd.notna(row['Radius_mm']) else None
    base_radius_px = int(radius_mm * MM_TO_PX) if radius_mm is not None else 30

    if cx is not None and cy is not None:
        coordinate_lookup[img_id] = {
            'cx': cx, 'cy': cy,
            'radius_mm': radius_mm,
            'base_radius_px': base_radius_px,
            'rating': row['Nodule_Rating'] if pd.notna(row['Nodule_Rating']) else None,
            'age': row['Age'] if pd.notna(row['Age']) else None,
            'gender': row['Gender'] if pd.notna(row['Gender']) else None,
            'diagnosis': row.get('Diagnosis', None),
            'location': row.get('Location', None),
        }

print(f"Loaded coordinates for {len(coordinate_lookup)} nodule profiles.")

# =========================================================================
# 2. Compute subband-entropy clutter over each target
# =========================================================================
results = []
processed_count = 0
failed_images = []

for img_id, info in coordinate_lookup.items():
    radius_mm = info['radius_mm']

    if FILTER_BY_SIZE:
        if radius_mm is None:
            continue  # no size info -> skip when filtering
        if MIN_RADIUS_MM is not None and radius_mm < MIN_RADIUS_MM:
            continue
        if MAX_RADIUS_MM is not None and radius_mm > MAX_RADIUS_MM:
            continue

    img_path = os.path.join(IMAGE_FOLDER, img_id)
    
    cx, cy = info['cx'], info['cy']
    base_radius = info['base_radius_px']

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        failed_images.append(img_id)
        continue

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR; grayscale X-rays load as 3 equal channels
    if img is None:
        print(f"Failed to read: {img_id}")
        failed_images.append(img_id)
        continue

    try:
        node_region = circular_mask(img.shape, cx, cy, base_radius)
        all_region = circular_mask(img.shape, cx, cy, base_radius + RING_DISTANCE_PX)
        peri_region = all_region & ~node_region

        node_crop = crop_to_region(img, node_region)
        peri_crop = crop_to_region(img, peri_region)
        all_crop = crop_to_region(img, all_region)

        results.append({
            'Image_ID': img_id,
            'Image_Path': img_path,
            'Nodule_Rating': info['rating'],
            'Radius_mm': info['radius_mm'],
            'Radius_px': base_radius,
            'Age': info['age'],
            'Gender': info['gender'],
            'X_Coord': cx,
            'Y_Coord': cy,
            'Diagnosis': info.get('diagnosis', ''),
            'Location': info.get('location', ''),
            'Local_Nodule_Clutter': subband_entropy_clutter(node_crop) if node_crop is not None else np.nan,
            'Local_Perinodular_Clutter': subband_entropy_clutter(peri_crop) if peri_crop is not None else np.nan,
            'Local_Total_Mask_Clutter': subband_entropy_clutter(all_crop) if all_crop is not None else np.nan,
        })
        processed_count += 1
        print(f"[{processed_count}/{len(coordinate_lookup)}] {img_id} "
              f"(Radius: {info['radius_mm']}mm / {base_radius}px) "
              f"Perinodular clutter: {results[-1]['Local_Perinodular_Clutter']:.4f}")

    except Exception as e:
        print(f"Error processing {img_id}: {e}")
        failed_images.append(img_id)

# =========================================================================
# 3. Save results
# =========================================================================
column_order = [
    'Image_ID', 'Image_Path', 'Nodule_Rating', 'R adius_mm', 'Radius_px',
    'Age', 'Gender', 'X_Coord', 'Y_Coord', 'Diagnosis', 'Location',
    'Local_Nodule_Clutter', 'Local_Perinodular_Clutter', 'Local_Total_Mask_Clutter',
]
df_out = pd.DataFrame(results)
if not df_out.empty:
    df_out = df_out[column_order]

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. Processed {processed_count}, failed {len(failed_images)}.")
print(f"Saved to {OUTPUT_CSV}")