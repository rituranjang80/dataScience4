import argparse
import cv2
import numpy as np
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--leaf', type=str, required=True, help='Path to transparent PNG of mango leaf')
    parser.add_argument('--background', type=str, required=True, help='Path to background image')
    parser.add_argument('--count', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='synthetic_output', help='Output directory')
    args = parser.parse_args()

    leaf = cv2.imread(args.leaf, cv2.IMREAD_UNCHANGED)
    bg = cv2.imread(args.background)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(args.count):
        # Random transformations
        scale = random.uniform(0.7, 1.3)
        angle = random.randint(-30, 30)
        
        # Rotate and scale
        M = cv2.getRotationMatrix2D((leaf.shape[1]/2, leaf.shape[0]/2), angle, scale)
        transformed = cv2.warpAffine(leaf, M, (leaf.shape[1], leaf.shape[0]))
        
        # Random position with boundary check
        max_x = bg.shape[1] - transformed.shape[1]
        max_y = bg.shape[0] - transformed.shape[0]
        x = random.randint(0, max(max_x, 0))
        y = random.randint(0, max(max_y, 0))
        
        # Composite
        result = bg.copy()
        alpha = transformed[:,:,3:4]/255.0
        result[y:y+transformed.shape[0], x:x+transformed.shape[1]] = \
            transformed[:,:,:3]*alpha + result[y:y+transformed.shape[0], x:x+transformed.shape[1]]*(1-alpha)
        
        cv2.imwrite(f'{args.output_dir}/synth_mango_{i}.jpg', result)

if __name__ == "__main__":
    main()

    # python generate_synthetic.py --leaf mango_leaf.png --background field.jpg --count 50 --output-dir synthetic_dataset