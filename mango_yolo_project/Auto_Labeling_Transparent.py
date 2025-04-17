import cv2
import numpy as np
import os
from ultralytics import YOLO

def transparent_to_mask(image_path):
    """Convert transparent PNG to binary mask"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Extract alpha channel (transparency)
    if img.shape[2] == 4:  # Check if image has alpha channel
        alpha = img[:,:,3]
        mask = (alpha > 0).astype(np.uint8) * 255
    else:
        # If no transparency, create mask of entire image
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    
    return mask

def mask_to_yolo_label(mask, output_txt_path, class_id=0):
    """Convert mask to YOLO format label file"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(output_txt_path, 'w') as f:
        for cnt in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Convert to YOLO format (normalized center coordinates and dimensions)
            height, width = mask.shape
            x_center = (x + w/2) / width
            y_center = (y + h/2) / height
            norm_width = w / width
            norm_height = h / height
            
            # Write to file
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

def auto_label_transparent_images(input_folder, class_id=0):
    """Process all transparent PNGs in a folder"""
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            
            # Skip if label already exists
            label_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(label_path):
                continue
            
            # Process image
            mask = transparent_to_mask(image_path)
            mask_to_yolo_label(mask, label_path, class_id)
            print(f"Created label: {label_path}")

# Example usage
auto_label_transparent_images("C:\\temp\\video\\video\\bgRemoveFile", class_id=0)