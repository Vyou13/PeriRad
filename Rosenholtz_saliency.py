import os
import re
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


# =============================================================================
# Saliency / visual distinctiveness analysis for pulmonary nodules
#
# This script compares the visual feature profile of a nodule region to the
# surrounding perinodular background using a Mahalanobis-style distance.
# The resulting scores are interpreted as saliency/distinctiveness values:
# higher values mean the nodule stands out more from its local background.
# =============================================================================


# =========================================================================
# CONFIGURATION
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"


CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "results", "clinical_data_clean.csv"),
    os.path.join(BASE_DIR, "clinical_data_clean.csv"),
]
CSV_PATH = next((path for path in CSV_CANDIDATES if os.path.exists(path)), CSV_CANDIDATES[0])
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "newRosenholtzSaliency10mm.csv")


RING_DISTANCE_PX = 57 # 10mm-57 expansion footprint 15mm- 86
MM_TO_PX = 5.7  # Conversion factor: 1mm = 5.7 pixels (based on 57px = 10mm)
# =========================================================================


def calculate_mahalanobis_clutter(target_vectors, bg_vectors):
    """Compute a Mahalanobis-style clutter score between a target region and its background."""
    if len(target_vectors) == 0 or len(bg_vectors) == 0:
        return np.nan


    T = np.mean(target_vectors, axis=0)
    mu_D = np.mean(bg_vectors, axis=0)


    Sigma_D = np.cov(bg_vectors, rowvar=False) + (np.eye(bg_vectors.shape[1]) * 1e-4)
    Sigma_D_inv = np.linalg.inv(Sigma_D)


    diff = T - mu_D
    return float(np.sqrt(max(0.0, diff.T @ Sigma_D_inv @ diff)))




def extract_rosenholtz_features(img_path):
    """Generates the 6 low-level visual feature maps specified in the paper."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_float = img.astype(np.float32) / 255.0
   
    lum = gaussian_filter(img_float, sigma=1)
    mean = gaussian_filter(img_float, sigma=2)
    sq_mean = gaussian_filter(img_float**2, sigma=2)
    contrast = np.sqrt(np.clip(sq_mean - mean**2, 0, None))                          
   
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_45 = grad_x + grad_y
    grad_135 = grad_x - grad_y
   
    return np.stack([lum, contrast, np.abs(grad_x), np.abs(grad_y), np.abs(grad_45), np.abs(grad_135)], axis=-1)


# =========================================================================
# 1. READ THE CLEAN CSV FILE
# =========================================================================
print("📖 Reading clean clinical data CSV...")
clinical_df = pd.read_csv(CSV_PATH)


# Display basic info
print(f"✅ Loaded {len(clinical_df)} nodule records")
print(f"📋 Columns: {', '.join(clinical_df.columns)}")


# Create lookup dictionary from the clean CSV
coordinate_lookup = {}


for idx, row in clinical_df.iterrows():
    # Get the image ID and convert to .png format
    img_id = row['Image_ID'].replace('.IMG', '.png')
   
    # Get coordinates and radius (in mm)
    cx = int(row['X_Coord']) if pd.notna(row['X_Coord']) else None
    cy = int(row['Y_Coord']) if pd.notna(row['Y_Coord']) else None
    radius_mm = row['Radius_mm'] if pd.notna(row['Radius_mm']) else None
   
    # Convert radius from mm to pixels
    if radius_mm is not None:
        base_radius_px = int(radius_mm * MM_TO_PX)
    else:
        base_radius_px = 30  # Default fallback if radius is missing
       
    # Get additional clinical info
    rating = row['Nodule_Rating'] if pd.notna(row['Nodule_Rating']) else None
    age = row['Age'] if pd.notna(row['Age']) else None
    gender = row['Gender'] if pd.notna(row['Gender']) else None
   
    if cx is not None and cy is not None:
        coordinate_lookup[img_id] = {
            'cx': cx,
            'cy': cy,
            'radius_mm': radius_mm,
            'base_radius_px': base_radius_px,
            'rating': rating,
            'age': age,
            'gender': gender,
            'diagnosis': row.get('Diagnosis', None),
            'location': row.get('Location', None)
        }


print(f"🎯 Successfully loaded coordinates for {len(coordinate_lookup)} nodule profiles!")


# =========================================================================
# 2. Process Rosenholtz Math Over the Found Targets
# =========================================================================
clutter_results = []
processed_count = 0
failed_images = []


for img_id, info in coordinate_lookup.items():
    img_path = os.path.join(IMAGE_FOLDER, img_id)
    cx, cy = info['cx'], info['cy']
    base_radius = info['base_radius_px']
    radius_mm = info['radius_mm']
   
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"⚠️ Image not found: {img_path}")
        failed_images.append(img_id)
        continue
   
    feature_maps = extract_rosenholtz_features(img_path)
    if feature_maps is None:
        print(f"⚠️ Failed to extract features from: {img_id}")
        failed_images.append(img_id)
        continue
       
    try:
        h, w, _ = feature_maps.shape
       
        # Create masks for nodule and surrounding ring
        blank_canvas_nodule = np.zeros((h, w), dtype=np.uint8)
        blank_canvas_dilated = np.zeros((h, w), dtype=np.uint8)
       
        # Use the radius from CSV for the nodule mask
        cv2.circle(blank_canvas_nodule, (cx, cy), base_radius, 255, -1)
        cv2.circle(blank_canvas_dilated, (cx, cy), base_radius + RING_DISTANCE_PX, 255, -1)
       
        mask_node_pixels = (blank_canvas_nodule > 0)
        mask_peri_pixels = (blank_canvas_dilated > 0) & (blank_canvas_nodule == 0)
        mask_all_pixels = (blank_canvas_dilated > 0)
       
        # Extract feature vectors
        target_vectors = feature_maps[mask_node_pixels]
        bg_vectors = feature_maps[mask_peri_pixels]
        all_mask_vectors = feature_maps[mask_all_pixels]
       
        # Skip if not enough pixels
        if len(target_vectors) == 0 or len(bg_vectors) == 0:
            print(f"⚠️ Not enough pixels in masks for {img_id} (nodule area too small?)")
            failed_images.append(img_id)
            continue
       
        # Whole-image feature vectors (used as the global background distribution)
        global_vectors = feature_maps.reshape(-1, feature_maps.shape[-1])


        # Calculate saliency metrics using the Mahalanobis-style distance.
        # Each column uses a distinct (target, background) pair:
        clutter_node = calculate_mahalanobis_clutter(target_vectors, global_vectors)      # nodule core vs WHOLE IMAGE
        clutter_perinodule = calculate_mahalanobis_clutter(bg_vectors, global_vectors)    # 15mm RING vs WHOLE IMAGE
        clutter_whole_thing = calculate_mahalanobis_clutter(all_mask_vectors, bg_vectors) # core+ring vs 15mm RING


        processed_count += 1
       
        # Extract case number from image ID
        match = re.search(r'\d+', img_id)
       
        # Store results with clinical info
        clutter_results.append({
            'Image_ID': img_id,
            'Image_Path': img_path,
            'Nodule_Rating': info['rating'],
            'Radius_mm': info['radius_mm'],
            'Radius_px': base_radius,  # Store the pixel radius used
            'Age': info['age'],
            'Gender': info['gender'],
            'X_Coord': cx,
            'Y_Coord': cy,
            'Diagnosis': info.get('diagnosis', ''),
            'Location': info.get('location', ''),
            'Local_Nodule_Saliency': clutter_node,
            'Local_Perinodular_Saliency': clutter_perinodule,
            'Local_Total_Mask_Saliency': clutter_whole_thing
        })
        print(f"✅ [{processed_count}/{len(coordinate_lookup)}] {img_id} "
              f"(Radius: {radius_mm}mm / {base_radius}px, Rating: {info['rating']}, "
              f"Clutter: {clutter_perinodule:.4f})")
       
    except Exception as e:
        print(f"❌ Error computing matrices on {img_id}: {e}")
        failed_images.append(img_id)


# =========================================================================
# 3. Save Results and Summary
# =========================================================================
df_out = pd.DataFrame(clutter_results)


# Reorder columns for better readability
column_order = [
     'Image_ID', 'Image_Path', 'Nodule_Rating', 'Radius_mm',
    'Radius_px', 'Age', 'Gender', 'X_Coord', 'Y_Coord', 'Diagnosis', 'Location',
    'Local_Nodule_Saliency', 'Local_Perinodular_Saliency', 'Local_Total_Mask_Saliency'
]
df_out = df_out[column_order]


# Save to CSV
df_out.to_csv(OUTPUT_CSV, index=False)


# Print summary
print(f"\n{'='*60}")
print(f"🎉 EXECUTION COMPLETE!")
print(f"{'='*60}")
print(f"✅ Successfully processed: {processed_count} images")
print(f"❌ Failed to process: {len(failed_images)} images")
if failed_images:
    print(f"   Failed images: {', '.join(failed_images[:5])}{'...' if len(failed_images) > 5 else ''}")


print(f"\n📊 Results saved to: {OUTPUT_CSV}")
print(f"📋 Total rows in output: {len(df_out)}")


# Show some basic statistics
print(f"\n📈 Summary Statistics:")
print("-" * 50)
if 'Nodule_Rating' in df_out.columns:
    print(f"Rating range: {df_out['Nodule_Rating'].min():.0f} - {df_out['Nodule_Rating'].max():.0f}")
    print(f"Mean rating: {df_out['Nodule_Rating'].mean():.2f}")
if 'Radius_mm' in df_out.columns:
    print(f"Radius (mm) range: {df_out['Radius_mm'].min():.1f} - {df_out['Radius_mm'].max():.1f}")
    print(f"Mean radius: {df_out['Radius_mm'].mean():.1f} mm")
if 'Radius_px' in df_out.columns:
    print(f"Radius (px) range: {df_out['Radius_px'].min():.0f} - {df_out['Radius_px'].max():.0f}")
    print(f"Mean radius: {df_out['Radius_px'].mean():.1f} px")
if 'Local_Perinodular_Saliency' in df_out.columns:
    print(f"Saliency range: {df_out['Local_Perinodular_Saliency'].min():.4f} - {df_out['Local_Perinodular_Saliency'].max():.4f}")
    print(f"Mean saliency: {df_out['Local_Perinodular_Saliency'].mean():.4f}")


# Show relationship between radius and rating
if 'Radius_mm' in df_out.columns and 'Nodule_Rating' in df_out.columns:
    correlation = df_out['Radius_mm'].corr(df_out['Nodule_Rating'])
    print(f"\n📈 Correlation between Radius and Rating: {correlation:.3f}")
    print(f"   (Positive correlation = larger nodules are easier to spot)")


if 'Local_Perinodular_Saliency' in df_out.columns and 'Nodule_Rating' in df_out.columns:
    correlation = df_out['Local_Perinodular_Saliency'].corr(df_out['Nodule_Rating'])
    print(f"📈 Correlation between Perinodular Saliency and Rating: {correlation:.3f}")
    print(f"   (Positive correlation = more distinct = easier to spot)")
    print(f"   (Negative correlation = less distinct = harder to spot)")


print(f"\n💡 Key insights:")
print(f"   - Base radius is now patient-specific from the CSV file")
print(f"   - Conversion factor: 1mm = {MM_TO_PX} pixels (based on 10m = 57px)")
print(f"   - Lower Local_Perinodular_Saliency = Less Distinct = Harder to Spot")
print(f"   - Higher Local_Perinodular_Saliency = More Distinct = Easier to Spot")



