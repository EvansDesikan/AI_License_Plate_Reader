import os
import xml.etree.ElementTree as ET
import shutil
import random

# CONFIG
RAW_IMAGES = 'dataset/images'
RAW_LABELS = 'dataset/annotations'
DATASET_DIR = 'yolo_dataset'

# Create folders: train/images, train/labels, val/images, val/labels
for split in ['train', 'val']:
    os.makedirs(f'{DATASET_DIR}/{split}/images', exist_ok=True)
    os.makedirs(f'{DATASET_DIR}/{split}/labels', exist_ok=True)

# Function to convert XML to YOLO (normalized 0-1 coordinates)
def convert_annotation(xml_file, width, height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []
    
    for obj in root.findall('object'):
        if obj.find('name').text == 'licence': # The class name in xml
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Math to normalize
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            labels.append(f"0 {x_center} {y_center} {w} {h}") # 0 is the class ID for plate
            
    return labels

# Loop through all images
files = [f for f in os.listdir(RAW_IMAGES) if f.endswith('.png')]
random.shuffle(files)
split_idx = int(len(files) * 0.8) # 80% train, 20% validation

print("⏳ Converting labels and organizing files...")

for i, filename in enumerate(files):
    split = 'train' if i < split_idx else 'val'
    
    # Copy Image
    src_img = os.path.join(RAW_IMAGES, filename)
    dst_img = os.path.join(DATASET_DIR, split, 'images', filename)
    shutil.copy(src_img, dst_img)
    
    # Convert Label
    xml_name = filename.replace('.png', '.xml')
    src_xml = os.path.join(RAW_LABELS, xml_name)
    
    # Read image size quickly to normalize
    import cv2
    img = cv2.imread(src_img)
    h, w, _ = img.shape
    
    yolo_labels = convert_annotation(src_xml, w, h)
    
    # Save Label file
    txt_name = filename.replace('.png', '.txt')
    with open(os.path.join(DATASET_DIR, split, 'labels', txt_name), 'w') as f:
        f.write('\n'.join(yolo_labels))

print("✅ Data Preparation Complete! Folder 'yolo_dataset' created.")