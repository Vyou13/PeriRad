import os
import re
import cv2
import numpy as np

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"  
TXT_FILE_PATH = r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt"         
OUTPUT_FOLDER = "images_with_edge_detection2"   # Folder where edge-detected images will be saved
IMAGE_NAME = '_edge_detection2.png'
# JSRT Standard Specifications
PIXEL_SPACING_MM = 0.175
RING_DISTANCE_PX = 57  # Distance in pixels from the inner circle to the outer circle (the ring region)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 1. Parse the JSRT coordinates text file using Regex 
with open(TXT_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

pattern = r'(JPCLN\d{3}\.IMG)'
parts = re.split(pattern, content)
records = []

for i in range(1, len(parts), 2):
    img_name_img_ext = parts[i]  # This is the original 'JPCLNxxx.IMG' filename 
    data_str = parts[i+1].strip()
    tokens = data_str.split()
    
    if len(tokens) >= 6:
        try:
            diameter_mm = float(tokens[1])  # Nodule max diameter in mm 
            cx = int(tokens[4])             # X coordinate center 
            cy = int(tokens[5])             # Y coordinate center 
            
            # Convert mm diameter to pixel radius
            radius_px = int((diameter_mm / 2) / PIXEL_SPACING_MM)
            
            # Map the database record to your actual .png file extension
            png_name = img_name_img_ext.replace('.IMG', '.png')
            
            records.append({'name': png_name, 'cx': cx, 'cy': cy, 'r': radius_px})
        except ValueError:
            continue

print(f"Parsed {len(records)} records. Calculating edge detection on PNG images...")

# 2. Process PNG images and overlay edge detection with rings
processed_count = 0
for idx, record in enumerate(records):
    img_path = os.path.join(IMAGE_FOLDER, record['name'])
    
    # Check if the PNG file actually exists in your folder
    if not os.path.exists(img_path):
        if processed_count < 3:  # Debug first few
            print(f"DEBUG: Not found: {record['name']}")
        continue
        
    # Read the PNG image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
    if img is None:
        print(f"Failed to read image: {record['name']}")
        continue

    # --- EDGE DETECTION PROCESSING ---
    
    # Calculate gradients using Sobel filter
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate edge magnitude
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize edge magnitude to 0-255 for display (grayscale - black and white)
    edge_magnitude_normalized = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Convert grayscale to BGR so we can draw colored circles
    edge_display = cv2.cvtColor(edge_magnitude_normalized, cv2.COLOR_GRAY2BGR)
    
    # Overlay circles on the edge detection image
    # Draw inner circle boundary in white
    # TODO: Change circle color by modifying the BGR tuple (e.g., (0, 0, 255) for red, (0, 255, 0) for green, (255, 0, 0) for blue)
    cv2.circle(edge_display, (record['cx'], record['cy']), record['r'], (255, 255, 255), 2)
    
    # Draw outer circle boundary in white
    outer_radius = record['r'] + RING_DISTANCE_PX
    cv2.circle(edge_display, (record['cx'], record['cy']), outer_radius, (255, 255, 255), 2)
    
    # Optional: Draw a small white dot at center
    cv2.circle(edge_display, (record['cx'], record['cy']), 3, (255, 255, 255), -1)
    
    # Save the edge detection visualization
    output_filename = record['name'].replace('.png', IMAGE_NAME)
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, edge_display)
    processed_count += 1

print(f"\nFinished! Processed {processed_count} images.")
print(f"Edge detection visualizations saved to: {OUTPUT_FOLDER}")
