# Mango Leaf YOLOv8 Automatic Trainer with Auto-Labeling

## Features
- Automatically generates YOLO-format labels for your mango leaf images
- Creates proper dataset structure
- Trains YOLOv8 model end-to-end

## Requirements
- Python 3.7 or later
- NVIDIA GPU recommended (for faster training)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place all your mango leaf images in the folder specified in `config.yaml`
2. Run the training:
```bash
python auto_train.py
```

## How Auto-Labeling Works
The script will:
1. Detect each image in your folder
2. Create a corresponding .txt file for each image
3. Generate a bounding box that:
   - Is centered in the image
   - Covers most of the image (with configurable margin)
   - Uses class ID 0 (mango_leaf)

## Configuration
Edit `config.yaml` to adjust:
- `image_folder`: Path to your images
- `label_margin`: Margin around auto-generated bounding box (0-0.5)
- Training parameters (epochs, batch size, etc.)

## Results
- Trained model saved in `runs/detect/auto_mango_leaf/`
- Automatic labels saved with original images