from ultralytics import YOLO
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

# 1. SETUP
# Load your custom trained model
print("⏳ Loading YOLO model...")
model = YOLO('best.pt') 

# Load the OCR Reader
print("⏳ Loading OCR engine...")
reader = easyocr.Reader(['en'])

# 2. RUN INFERENCE
img_path = 'car.jpg' # You can change this to any image path
image = cv2.imread(img_path)

# Run YOLO on the image
results = model(image)

# 3. PROCESS RESULTS
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = float(box.conf[0])
        
        print(f"✅ Plate detected! Confidence: {confidence:.2f}")

        # 4. CROP THE PLATE
        # Add a small buffer/padding if you want, but raw crop works too
        plate_crop = image[y1:y2, x1:x2]

        # 5. READ TEXT (OCR)
        # We convert to gray for easier reading
        gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        ocr_result = reader.readtext(gray_plate)
        
        if len(ocr_result) > 0:
            text = ocr_result[0][-2]
            print(f"🚗 LICENSE PLATE: {text}")

            # 6. DRAW ON IMAGE
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
            
            # Show the crop and the full result
            plt.figure(figsize=(10,5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
            plt.title("YOLO Crop")
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"Result: {text}")
            
            plt.show()
        else:
            print("❌ Detected plate, but OCR could not read text.")

print("🏁 Finished.")