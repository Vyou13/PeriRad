#does edge detection on the image and saves the results to a csv file
import os
import re
import cv2
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"  
CSV_PATH = r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt"         
OUTPUT_CSV = "nodule_edge_results10mm1.csv" 
# ---------------------
# JSRT Standard: Images are 2048x2048 pixels. 
# Each pixel represents exactly 0.175 mm.
PIXEL_SPACING_MM = 0.175 
# 10 mm ring zone is 57 pixels wide
RING_DISTANCE_PX = 57  
# ---------------------

# 1. Read the raw text file content
with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 2. Use regex to split text right at the image name boundary
pattern = r'(JPCLN\d{3}\.IMG)'
parts = re.split(pattern, content)

records = []

# Loop through the regex parts to parse data tokens
for i in range(1, len(parts), 2):
    img_name = parts[i]             # e.g., 'JPCLN001.IMG'
    data_str = parts[i+1].strip()   # The metadata string following it
    tokens = data_str.split()
    
    if len(tokens) >= 6:
        try:
            clarity = int(tokens[0])       # Clarity level (0-5 scale)
            diameter_mm = float(tokens[1]) # Nodule max diameter in mm
            cx = int(tokens[4])            # X coordinate center
            cy = int(tokens[5])            # Y coordinate center
            
            # Calculate the pure radius in pixels without buffer
            radius_px = int((diameter_mm / 2) / PIXEL_SPACING_MM)
            
            records.append({
                'image_name': img_name,
                'cx': cx,
                'cy': cy,
                'radius_px': radius_px,
                'diameter_mm': diameter_mm,
                'clarity': clarity
            })
        except ValueError:
            continue

print(f"Successfully parsed {len(records)} nodule positions from metadata.")
print("Beginning image edge-detection and sharpness analysis (10mm configuration)...")

# 3. Process every image automatically
results = []
for record in records:
    img_name = record['image_name']
    
    # Convert .IMG to .png since actual files are in .png format
    img_name_png = img_name.replace('.IMG', '.png')
    img_path = os.path.join(IMAGE_FOLDER, img_name_png)
    
    # Load image in raw grayscale format
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Skipping {img_name}: File not found in image folder.")
        continue
        
    # --- EDGE DETECTION PROCESSING ---
    
    # Calculate gradients across the X and Y plane using a Sobel filter
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Mathematical edge strength (magnitude) at every coordinate
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Create an image-sized blank mask for isolating the ring region around the nodule
    edge_mask = np.zeros(img.shape, dtype=np.uint8)
    
    # Draw the outer circle in white
    outer_radius = record['radius_px'] + RING_DISTANCE_PX
    cv2.circle(edge_mask, (record['cx'], record['cy']), outer_radius, 255, -1)
    
    # Draw the inner circle (to exclude) - overlay with black
    cv2.circle(edge_mask, (record['cx'], record['cy']), record['radius_px'], 0, -1)
    
    # Extract edge values from the ring region (between inner and outer circles)
    ring_edge_pixels = edge_magnitude[edge_mask == 255]
    
    if len(ring_edge_pixels) > 0:
        avg_edge_sharpness = np.mean(ring_edge_pixels)
        max_edge_sharpness = np.max(ring_edge_pixels)
    else:
        avg_edge_sharpness = np.nan
        max_edge_sharpness = np.nan
        
    results.append({
        "image_name": img_name,
        "center_x": record['cx'],
        "center_y": record['cy'],
        "boundary_radius_pixels": record['radius_px'],
        "diameter_mm": record['diameter_mm'],
        "clarity": record['clarity'],
        "average_edge_sharpness": round(avg_edge_sharpness, 2),
        "maximum_edge_sharpness": round(max_edge_sharpness, 2)
    })

# 4. Export all calculations to your new edge spreadsheet
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

# After calculating edge_magnitude, you could:
# 1. Normalize edge magnitude to 0-255 for display
edge_magnitude_normalized = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 2. Apply colormap to see intensity
edge_colored = cv2.applyColorMap(edge_magnitude_normalized, cv2.COLORMAP_JET)

# 3. Save the visualization
output_path = os.path.join("edge_visualizations", img_name_png)
cv2.imwrite(output_path, edge_colored)

print(f"\nFinished! Edge analysis data saved to: {OUTPUT_CSV}")