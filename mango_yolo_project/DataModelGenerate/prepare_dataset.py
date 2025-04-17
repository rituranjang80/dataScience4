import yaml
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_yolo_dataset(labeled_folder, output_dir="training"):
    """Prepare YOLOv8 dataset structure"""
    # Create folders
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    # Get all labeled images
    images = [f for f in os.listdir(labeled_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split train/val (80/20)
    train, val = train_test_split(images, test_size=0.2, random_state=42)
    
    # Copy files
    for name, files in [('train', train), ('val', val)]:
        for file in files:
            # Copy image
            shutil.copy(
                os.path.join(labeled_folder, file),
                os.path.join(output_dir, 'images', name, file)
            )
            # Copy label
            label_file = os.path.splitext(file)[0] + '.txt'
            shutil.copy(
                os.path.join(labeled_folder, label_file),
                os.path.join(output_dir, 'labels', name, label_file)
            )
    
    # Create dataset.yaml
    yaml_content = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'mango_leaf'}
    }
    
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"Dataset prepared in {output_dir}")

prepare_yolo_dataset("labeled_data")