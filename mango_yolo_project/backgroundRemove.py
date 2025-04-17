import os
from rembg import remove
from PIL import Image

def process_images(source_root, target_root):
    """
    Process all images in source_root and subfolders, save to target_root with same structure
    """
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create paths
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_root)
                target_dir = os.path.join(target_root, relative_path)
                target_path = os.path.join(target_dir, file)
                
                # Create target directory if needed
                os.makedirs(target_dir, exist_ok=True)
                
                # Process image
                try:
                    with open(source_path, 'rb') as i:
                        with open(target_path, 'wb') as o:
                            input_img = i.read()
                            output_img = remove(input_img)
                            o.write(output_img)
                    print(f"Processed: {source_path} -> {target_path}")
                except Exception as e:
                    print(f"Error processing {source_path}: {str(e)}")

# Configuration
source_folder = r"C:\\temp\\video\\video\\archive1"
target_folder = r"C:\\temp\\video\\video\\mango leaf_bgRemoveFile"

# Run processing
process_images(source_folder, target_folder)
print("Background removal complete!")