import cv2
from ultralytics import YOLO

def detect_mango_leaves_in_video(video_path, model_path, output_path='output.mp4', conf_threshold=0.5):
    """
    Process a video file to detect mango leaves
    
    Args:
        video_path (str): Path to input video
        model_path (str): Path to trained YOLOv8 model (.pt file)
        output_path (str): Path to save output video
        conf_threshold (float): Confidence threshold (0-1)
    """
    # Load the trained model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties for output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection
        results = model.predict(frame, conf=conf_threshold)
        
        # Annotate frame with detections
        annotated_frame = results[0].plot()
        cv2.imwrite('C:\\temp\\video\\video\\output.jpg', annotated_frame)

        # Write frame to output
        out.write(annotated_frame)
        

        # # Display live preview (optional)
        # cv2.imshow('Mango Leaf Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    # Example usage - change these paths
    input_video = "C:\\temp\\video\\video\\\dog.mp4"
    trained_model = "C:\\temp\\video\\training\\mango_yolo_project\\DataModelGenerate\\runs\\detect\\mango_leaf_v12\\weights\\last.pt"#"runs/detect/auto_mango_leaf/weights/best.pt"
    
    detect_mango_leaves_in_video(
        video_path=input_video,
        model_path=trained_model,
        output_path="C:\\temp\\video\\video\\mango1_output512.mp4",
        conf_threshold=0.5
    )