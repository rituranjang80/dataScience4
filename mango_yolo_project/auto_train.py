import os
import yaml
import cv2
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import shutil

def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

def generate_yolo_labels(image_path, config):
    """Generate a YOLO-format label file for an image"""
    # Read image dimensions
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    height, width = img.shape[:2]
    
    # Create bounding box (centered with margin)
    margin = config['label_margin']
    x_center = 0.5
    y_center = 0.5
    box_width = 1.0 - (2 * margin)
    box_height = 1.0 - (2 * margin)
    
    # Create label content
    label_content = f"{config['class_id']} {x_center} {y_center} {box_width} {box_height}"
    
    # Write label file
    label_path = os.path.splitext(image_path)[0] + '.txt'
    with open(label_path, 'w') as f:
        f.write(label_content)
    
    return True

def prepare_dataset(config):
    """Prepare dataset with auto-generated labels"""
    # Create dataset structure
    dataset_dir = "mango_yolo_dataset"
    os.makedirs(f"{dataset_dir}/images/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/images/val", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labels/val", exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(config['image_folder']) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Generate labels for all images
    print("\nGenerating YOLO labels for images...")
    for img_file in image_files:
        img_path = os.path.join(config['image_folder'], img_file)
        if not generate_yolo_labels(img_path, config):
            print(f"Warning: Could not process {img_file}")
    
    # Split into train/val
    train_files, val_files = train_test_split(image_files, train_size=config['train_ratio'])
    
    # Copy files to YOLO structure
    for phase, files in [('train', train_files), ('val', val_files)]:
        for file in files:
            # Copy image
            shutil.copy(f"{config['image_folder']}/{file}", 
                       f"{dataset_dir}/images/{phase}/{file}")
            # Copy label
            label_file = f"{os.path.splitext(file)[0]}.txt"
            shutil.copy(f"{config['image_folder']}/{label_file}", 
                       f"{dataset_dir}/labels/{phase}/{label_file}")
    
    # Create YAML config
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: config['class_name']}
    }
    
    with open(f"{dataset_dir}/dataset.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
    
    return True

def train_model(config):
    """Train YOLOv8 model"""
    model = YOLO(config['model_type'])
    results = model.train(
        data="mango_yolo_dataset/dataset.yaml",
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        name='auto_mango_leaf',
        augment=config['augment'],
        hsv_h=config['hsv_h'],
        hsv_s=config['hsv_s'],
        hsv_v=config['hsv_v']
    )
    return results

if __name__ == "__main__":
    config = load_config()
    if prepare_dataset(config):
        print("\nDataset prepared successfully. Starting training...")
        train_model(config)
        print("\nTraining completed! Results saved in runs/detect/auto_mango_leaf")