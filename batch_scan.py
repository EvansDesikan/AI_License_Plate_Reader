import cv2
import numpy as np
import imutils
import easyocr
import os
import matplotlib.pyplot as plt

# 1. SETUP
IMAGE_DIR = 'dataset/images'  # <--- Matches your new folder structure
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("⏳ Initializing OCR Engine...")
reader = easyocr.Reader(['en'])

# 2. LOOP THROUGH IMAGES
files = os.listdir(IMAGE_DIR)

# We will scan the first 5 images just to test
for file_name in files[:5]: 
    img_path = os.path.join(IMAGE_DIR, file_name)
    print(f"\n📸 Processing: {file_name}")
    
    img = cv2.imread(img_path)
    if img is None: continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. DETECT PLATE
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    edged = cv2.Canny(bfilter, 30, 200) 
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        print("   ❌ No plate contour found.")
        continue

    # 4. CROP & READ
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # Read the text
    result = reader.readtext(cropped_image)
    
    if len(result) > 0:
        text = result[0][-2]
        print(f"   🚗 DETECTED: {text}")

        # Draw green box and text
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1]+60), 
                          fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)

        # Save result
        save_path = os.path.join(OUTPUT_DIR, f"detected_{file_name}")
        cv2.imwrite(save_path, res)
        print(f"   ✅ Saved to {save_path}")
    else:
        print("   ❌ Plate found, but text was too blurry.")

print("\n🏁 Done! Open the 'results' folder to see the magic.")