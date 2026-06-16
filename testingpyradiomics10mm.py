import os
import re
import cv2
import numpy as np
import pandas as pd
from radiomics import featureextractor

# --- CONFIGURATION ---
IMAGE_FOLDER = r"C:\Users\vivia\Downloads\JSRT images with nodules (154 images)\Radiology"  
CSV_PATH = r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt"         

# Folders and files this script will handle
MASK_OUTPUT_FOLDER = r"C:\Users\vivia\OneDrive\Desktop\lab\generated_masks"
BATCH_INPUT_CSV = "pyradiomics_batch_input.csv"
FINAL_OUTPUT_CSV = "nodule_perinodular_pyradiomics_results10mm.csv" 

PIXEL_SPACING_MM = 0.175 
RING_DISTANCE_PX = 57  # 10mm region width mapping to 57 pixels
# ---------------------

os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)

# =========================================================================
# STEP 1: PARSE TEXT FILE & GENERATE THE BATCH CSV LAYOUT
# =========================================================================
print("1. Parsing metadata text file and drawing physical mask images...")

with open(CSV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

pattern = r'(JPCLN\d{3}\.IMG)'
parts = re.split(pattern, content)
batch_rows = []

for i in range(1, len(parts), 2):
    img_name = parts[i]
    data_str = parts[i+1].strip()
    tokens = data_str.split()
    
    if len(tokens) >= 6:
        try:
            clarity = int(tokens[0])
            diameter_mm = float(tokens[1]) 
            cx = int(tokens[4])            
            cy = int(tokens[5])            
            radius_px = int((diameter_mm / 2) / PIXEL_SPACING_MM)
            
            # Check for extension variations (.png vs .PNG)
            img_name_png = img_name.replace('.IMG', '.png')
            img_path = os.path.join(IMAGE_FOLDER, img_name_png)
            if not os.path.exists(img_path):
                img_name_png = img_name.replace('.IMG', '.PNG')
                img_path = os.path.join(IMAGE_FOLDER, img_name_png)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                # Initialize a blank mask
                mask = np.zeros(img.shape, dtype=np.uint8)
                outer_radius = radius_px + RING_DISTANCE_PX
                
                # =========================================================================
                # MASK MAPPING OPTIONS (Uncomment only ONE of the three options below)
                # =========================================================================
                
                # --- OPTION A: PERINODULAR RING ONLY (Your Current Settings) ---
                cv2.circle(mask, (cx, cy), outer_radius, 1, -1) 
                cv2.circle(mask, (cx, cy), radius_px, 0, -1)   
                
                # --- OPTION B: NODULE AREA ONLY ---
                # # To extract features only from the target nodule itself:
                # cv2.circle(mask, (cx, cy), radius_px, 1, -1)
                
                # --- OPTION C: BOTH AREAS TOGETHER (Full Nodule + 10mm Perinodular Ring) ---
                # # To extract features from the nodule and its surrounding context combined:
                # cv2.circle(mask, (cx, cy), outer_radius, 1, -1)
                
                # --- THE VISIBILITY FIX ---
                # We multiply by 255 ONLY when saving to your folder so you can visually see it.
                visible_mask = mask * 255
                
                # Save mask file
                mask_filename = f"{img_name.replace('.IMG', '')}_ring_mask.png"
                mask_path = os.path.join(MASK_OUTPUT_FOLDER, mask_filename)
                cv2.imwrite(mask_path, visible_mask) 
                
                # Keep extra metadata tracking columns to match the output requirements
                batch_rows.append({
                    "Image": img_path,
                    "Mask": mask_path,
                    "Label": 255,  # Changed to 255 so PyRadiomics looks at the bright white ring
                    "Clarity": clarity,
                    "Diameter_mm": diameter_mm
                })
        except Exception:
            continue

# Create and save the blueprint CSV
df_batch = pd.DataFrame(batch_rows)
df_batch.to_csv(BATCH_INPUT_CSV, index=False)
print(f"--> Success! Created {len(batch_rows)} rows inside '{BATCH_INPUT_CSV}'")

# =========================================================================
# STEP 2: CUSTOM MANUAL LOOP PROCESSING
# =========================================================================
print("\n2. Initializing PyRadiomics Feature Extractor...")
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('ngtdm')

print("3. Iterating through batch input rows to extract radiomics...")
final_results = []

# Read the batch file we just created
input_data = pd.read_csv(BATCH_INPUT_CSV)

for idx, row in input_data.iterrows():
    img_p = row['Image']
    msk_p = row['Mask']
    lbl = int(row['Label'])
    
    try:
        # Pass the file paths and extraction label exactly as requested
        feature_vector = extractor.execute(img_p, msk_p, label=lbl)
        
        # Strip background pipeline logs
        clean_features = {k: v for k, v in feature_vector.items() if not k.startswith('diagnostics')}
        
        # Copy original batch info columns to append data next to it
        for col in input_data.columns:
            clean_features[col] = row[col]
            
        final_results.append(clean_features)
        print(f"   ✅ Processed: {os.path.basename(img_p)}")
        
    except Exception as e:
        print(f"   ❌ Failed to extract features for {os.path.basename(img_p)}: {e}")
        continue

# =========================================================================
# STEP 3: EXPORT RESULTS
# =========================================================================
if len(final_results) == 0:
    print("\n❌ Error: No features were extracted. Check your file pathways.")
else:
    df_output = pd.DataFrame(final_results)
    
    # Place original metadata columns up front, calculations append next to them
    original_cols = list(input_data.columns)
    calc_cols = [c for c in df_output.columns if c not in original_cols]
    df_output = df_output[original_cols + calc_cols]
    
    df_output.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"\n Finished! Comprehensive database exported to: {FINAL_OUTPUT_CSV}")