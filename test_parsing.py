import re

TXT_FILE_PATH = r"C:\Users\vivia\Downloads\Clinical_Information\Clinical_Information\CLNDAT_EN.txt"

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
            
            # Map the database record to your actual .png file extension
            png_name = img_name_img_ext.replace('.IMG', '.png')
            
            records.append({'name': png_name, 'cx': cx, 'cy': cy})
        except ValueError as e:
            print(f"Error parsing {img_name_img_ext}: {e}")
            continue

print(f"Parsed {len(records)} records")
print(f"First 5 records: {records[:5]}")
