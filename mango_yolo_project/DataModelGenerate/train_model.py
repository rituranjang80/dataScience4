import shutil
from ultralytics import YOLO
import os

def train_yolov8():
    # Load model
    model = YOLO('yolov8n.pt')  # Load pretrained
    
    # Train
    results = model.train(
        data='training/dataset.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        device='cpu',  # Use GPU (set to 'cpu' if no GPU)
        name='mango_leaf_v1',
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5
    )
    
    # Export best model
    best_model = YOLO('runs/detect/mango_leaf_v1/weights/best.pt')
    os.makedirs('trained_models', exist_ok=True)
    best_model.export(format='onnx', simplify=True, imgsz=640)
    shutil.move('runs/detect/mango_leaf_v1/weights/best.onnx', 'trained_models/mango_leaf.onnx')
    shutil.move('runs/detect/mango_leaf_v1/weights/best.pt', 'trained_models/mango_leaf.pt')

train_yolov8()