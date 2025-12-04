import cv2
import numpy as np
import imutils
import easyocr
import matplotlib.pyplot as plt

# 1. READ IMAGE
img_name = 'car.jpg'
img = cv2.imread(img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. FILTER & EDGE DETECTION
# Noise reduction
bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
# Canny Edge Detection
edged = cv2.Canny(bfilter, 30, 200) 

# 3. FIND CONTOURS (Looking for rectangles)
keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    # Approximate the shape
    approx = cv2.approxPolyDP(contour, 10, True)
    # If it has 4 corners, it's likely our plate
    if len(approx) == 4:
        location = approx
        break

if location is None:
    print("Could not find the license plate!")
else:
    print("✅ License Plate Contour Found")

    # 4. MASKING (Isolate the plate)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Crop the image to just the plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    # 5. READ TEXT (OCR)
    print("⏳ Reading text... (This downloads the model first time)")
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)
    
    if len(result) > 0:
        text = result[0][-2]
        print(f"🚗 LICENCE PLATE: {text}")

        # 6. VISUALIZE
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1]+60), 
                          fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
        res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
        
        # Show using Matplotlib (because cv2.imshow sometimes crashes with big OCR models)
        plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected: {text}")
        plt.show()
    else:
        print("❌ Could not read text from the plate.")