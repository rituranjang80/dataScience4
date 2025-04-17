from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np

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
    
    # Initialize lists to store metrics
    all_results = []
    class_names = ['mango_leaf', 'other_leaf']  # Update with your class names
    
    # Evaluate on each image
    for img_path in test_images:
        results = model.predict(img_path, conf=conf_threshold)
        all_results.extend(results)
    #   "C:\temp\video\training\mango_yolo_project\DataModelGenerate\training\dataset.yaml
    path="C:\\temp\\video\\training\\mango_yolo_project\\DataModelGenerate\\training\\dataset.yaml"
    # Calculate metrics
    metrics = model.val(data=path,  # Path to your dataset YAML
                       batch=8,
                       imgsz=640,
                       conf=conf_threshold,
                       iou=0.6)
    
    # Print summary metrics
    print("\nEvaluation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    # print(f"Precision: {metrics.box.precision.mean():.4f}")
    # print(f"Recall: {metrics.box.recall.mean():.4f}")
    
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
    for i, result in enumerate(all_results[:3]):  # Show first 3 results
        result.save(filename=f'prediction_{i}.jpg')
        plotted = result.plot()
        plt.imshow(plotted)
        plt.title(f'Prediction Example {i+1}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    # Configure these paths
    model_path = "C:\\temp\\video\\training\\mango_yolo_project\\DataModelGenerate\\runs\\detect\\mango_leaf_v12\\weights\\last2.pt"#"runs/detect/auto_mango_leaf/weights/best.pt"
    test_images_dir = "C:\\temp\\video\\video\\test"
    
    evaluate_model(
        model_path=model_path,
        test_images_dir=test_images_dir,
        conf_threshold=0.5
    )