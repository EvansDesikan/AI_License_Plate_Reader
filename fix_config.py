import os

# 1. Get the absolute path of your current folder
current_dir = os.getcwd().replace('\\', '/') # Windows fix

# 2. Define the correct dataset path
# We point directly to the 'yolo_dataset' folder we created earlier
dataset_path = f"{current_dir}/yolo_dataset"

# 3. Create the content for data.yaml
yaml_content = f"""path: {dataset_path}
train: train/images
val: val/images

nc: 1
names: ['license_plate']
"""

# 4. Overwrite the file
with open('data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"✅ Fixed data.yaml!")
print(f"📂 Dataset set to: {dataset_path}")
print("🚀 You can run training now.")