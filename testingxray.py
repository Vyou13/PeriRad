import os
import re
import cv2
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"  
CSV_PATH = r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt"         
OUTPUT_CSV = "nodule_brightness_results15mm.csv" 
# ---------------------
# JSRT Standard: Images are 2048x2048 pixels. 
# Each pixel represents exactly 0.175 mm.
PIXEL_SPACING_MM = 0.175 
#10 and 15 mm nodules would be 57 and 86 pixels in diameter
RADIUS_BUFFER_PX = 10  # Additional pixels to add to the radius for a more inclusive nodule area
RING_DISTANCE_PX = 86  # Distance in pixels from the inner circle to the outer circle (the ring region)
# ---------------------

# 1. Read the raw text file content
with open(r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt", 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 2. Use regex to split text right at the image name boundary
# This cleanly fixes the mashed-together records (e.g., 'matomaJPCLN003.IMG')
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
            
            # Dynamically calculate the perfect circle radius in pixels:
            # Radius = (Diameter / 2) / 0.175 mm per pixel
           
            radius_px = int((diameter_mm / 2) / PIXEL_SPACING_MM) + RADIUS_BUFFER_PX
            
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
print("Beginning image pixel-intensity analysis...")

# 3. Process every image automatically
results = []
for record in records:
    img_name = record['image_name']
    
    # Convert .IMG to .png since actual files are in .png format
    img_name_png = img_name.replace('.IMG', '.png')
    img_path = os.path.join(IMAGE_FOLDER, img_name_png)
    
    # Load image in raw grayscale format (keeps original brightness entirely intact)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Skipping {img_name}: File not found in image folder.")
        continue
        
    # Create an image-sized blank black mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    # Draw the outer circle (inner circle + ring distance) in white 
    outer_radius = record['radius_px'] + RING_DISTANCE_PX
    cv2.circle(mask, (record['cx'], record['cy']), outer_radius, 255, -1)
    
    # Draw the inner circle (to exclude from calculations) - overlay with black
    cv2.circle(mask, (record['cx'], record['cy']), record['radius_px'], 0, -1)
    
    # Extract the raw pixel values from only the ring region (between inner and outer circles)
    nodule_pixels = img[mask == 255]
    
    if len(nodule_pixels) > 0:
        avg_brightness = np.mean(nodule_pixels)
    else:
        avg_brightness = np.nan
        
    results.append({
        "image_name": img_name,
        "center_x": record['cx'],
        "center_y": record['cy'],
        "inner_radius_pixels": record['radius_px'],
        "outer_radius_pixels": record['radius_px'] + RING_DISTANCE_PX,
        "diameter_mm": record['diameter_mm'],
        "clarity": record['clarity'],
        "average_brightness": round(avg_brightness, 2)
    })

# 4. Export all calculations to a new spreadsheet
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)

print(f"\nFinished! Compiled data saved to: {OUTPUT_CSV}")