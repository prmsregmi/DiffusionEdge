"""
Convert BIPED ground truth PNG images to MAT format matching UDED structure.

UDED format (verified):
- groundTruth: shape (1, 1), dtype object
- groundTruth[0][0]: structured array with dtype [('Boundaries', 'O')]
- groundTruth[0][0]['Boundaries'][0,0]: 2D uint8 array (H, W) of 0/1 values
"""
import os
import cv2
import numpy as np
from scipy.io import savemat, loadmat


def create_gt_mat_uded_format(boundaries_array):
    """Create groundTruth structure matching exact UDED format."""
    # Inner structured array: dtype [('Boundaries', 'O')], shape (1, 1)
    inner = np.empty((1, 1), dtype=[('Boundaries', 'O')])
    inner['Boundaries'][0, 0] = boundaries_array
    
    # Outer object array: shape (1, 1)
    outer = np.empty((1, 1), dtype=object)
    outer[0, 0] = inner
    
    return {'groundTruth': outer}


def convert_biped_gt_to_mat(gt_dir, output_dir):
    """Convert all BIPED GT PNGs to MAT format."""
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for filename in sorted(os.listdir(gt_dir)):
        if not filename.lower().endswith('.png'):
            continue
        
        filepath = os.path.join(gt_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {filename}")
            continue
        
        # Convert to binary (0/1) - BIPED GT is typically 0 (bg) and 255 (edge)
        boundaries = (img > 127).astype(np.uint8)
        
        # Create MAT structure matching UDED format
        mat_data = create_gt_mat_uded_format(boundaries)
        
        out_name = os.path.splitext(filename)[0] + '.mat'
        out_path = os.path.join(output_dir, out_name)
        savemat(out_path, mat_data)
        count += 1
    
    print(f"Converted {count} files to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert GT edge PNGs to MAT format")
    parser.add_argument("--gt_dir", required=True, help="Directory with GT edge PNG images")
    parser.add_argument("--output_dir", required=True, help="Output directory for MAT files")
    args = parser.parse_args()
    convert_biped_gt_to_mat(args.gt_dir, args.output_dir)
    
    # Verify first output file
    mat_files = [f for f in os.listdir(args.output_dir) if f.endswith('.mat')]
    if mat_files:
        sample = loadmat(os.path.join(args.output_dir, mat_files[0]))
        boundaries = sample['groundTruth'][0][0]['Boundaries'][0, 0]
        print(f"\nVerified {mat_files[0]}: shape {boundaries.shape}, unique {np.unique(boundaries)}")
