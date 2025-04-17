import cv2
import numpy as np
import os
from tqdm import tqdm

def create_yolo_labels_from_transparent(image_folder, output_folder, class_id=0):
    """Generate YOLO labels from transparent PNGs"""
    os.makedirs(output_folder, exist_ok=True)
    
    for img_name in tqdm(os.listdir(image_folder), desc="Processing images"):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # Extract alpha channel (transparency)
        if img.shape[2] == 4:
            alpha = img[:,:,3]
            mask = (alpha > 0).astype(np.uint8) * 255
        else:
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        
        # Find contours and bounding boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare YOLO label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(output_folder, label_name)
        height, width = mask.shape
        
        with open(label_path, 'w') as f:
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Convert to YOLO format (normalized)
                x_center = (x + w/2) / width
                y_center = (y + h/2) / height
                norm_w = w / width
                norm_h = h / height
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # Copy original image to labeled folder
        cv2.imwrite(os.path.join(output_folder, img_name), img)

# Configuration
input_folder = "C:\\temp\\video\\video\\mango leaf_bgRemoveFile"
output_folder = "labeled_data"
class_id = 0  # 0 for mango_leaf

create_yolo_labels_from_transparent(input_folder, output_folder, class_id)