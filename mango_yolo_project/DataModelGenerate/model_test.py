from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import csv
import cv2
from datetime import datetime

def evaluate_model(model_path, test_images_dir, conf_threshold=0.5):
    """
    Evaluate YOLOv8 model on test images with different leaf types
    
    Args:
        model_path (str): Path to .pt model file
        test_images_dir (str): Directory containing test images
        conf_threshold (float): Confidence threshold for detection
    """
    # Load model
    model = YOLO(model_path)
    
    # Get all test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Initialize lists to store metrics and CSV data
    all_results = []
    csv_data = []
    class_names = ['mango_leaf', 'other_leaf']  # Update with your class names
    
    # Evaluate on each image
    for img_path in test_images:
        results = model.predict(img_path, conf=conf_threshold)
        all_results.extend(results)
        
        # Extract data for CSV
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                csv_data.append([
                    os.path.basename(img_path),
                    class_names[class_id],
                    confidence
                ])
    
    # Save to CSV
    with open('C:\\temp\\video\\video\\test\\leaf_detections.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Leaf Type', 'Confidence'])
        writer.writerows(csv_data)
    
    # Calculate metrics
    path = "C:\\temp\\video\\training\\mango_yolo_project\\DataModelGenerate\\training\\dataset.yaml"
    metrics = model.val(data=path,
                       batch=8,
                       imgsz=640,
                       conf=conf_threshold,
                       iou=0.6)
    
    # Print summary metrics
    print("\nEvaluation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    # Generate confusion matrix
    if hasattr(metrics, 'confusion_matrix'):
        conf_matrix = metrics.confusion_matrix.matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                    display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    # Visualize some predictions
    for i, result in enumerate(all_results[:3]):
        result.save(filename=f'prediction_{i}.jpg')
        plotted = result.plot()
        plt.imshow(plotted)
        plt.title(f'Prediction Example {i+1}')
        plt.axis('off')
        plt.show()

def detect_in_video(model_path, video_path, output_csv='video_detections.csv', conf_threshold=0.5):
    """
    Detect objects in video and save detections with timestamps
    
    Args:
        model_path (str): Path to .pt model file
        video_path (str): Path to input video
        output_csv (str): Path to output CSV file
        conf_threshold (float): Confidence threshold
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Frame', 'Leaf Type', 'Confidence'])
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate timestamp
            frame_count += 1
            current_time = frame_count / fps
            timestamp = str(datetime.utcfromtimestamp(current_time))#.split(' ')[1][:8])
            
            # Perform detection
            results = model.predict(frame, conf=conf_threshold)
            
            # Record detections
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    writer.writerow([
                        timestamp,
                        frame_count,
                        ['mango_leaf', 'other_leaf'][class_id],
                        confidence
                    ])
    
    cap.release()
    print(f"Video analysis complete. Results saved to {output_csv}")

if __name__ == "__main__":
    # Configure these paths
    model_path = "C:\\temp\\video\\training\\mango_yolo_project\\DataModelGenerate\\runs\\detect\\mango_leaf_v12\\weights\\last2.pt"
    test_images_dir = "C:\\temp\\video\\video\\test"
    video_path = "C:\\temp\\video\\video\\test\\sofa.mp4"
    
    # 1. Evaluate on test images
    evaluate_model(
        model_path=model_path,
        test_images_dir=test_images_dir,
        conf_threshold=0.5
    )
    
    # # 2. Analyze video
    # detect_in_video(
    #     model_path=model_path,
    #     video_path=video_path,
    #     output_csv='C:\\temp\\video\\video\\test\\video_detections.csv',
    #     conf_threshold=0.5
    # )