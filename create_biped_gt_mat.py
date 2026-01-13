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
    gt_dir = "data/BIPED/gt"
    output_dir = "data/BIPED/gt_mat"
    convert_biped_gt_to_mat(gt_dir, output_dir)
    
    # Verify by comparing with UDED format
    print("\n--- Verification ---")
    
    uded = loadmat('data/UDED/gt_mat/12-cameraman.mat')
    print(f"UDED groundTruth shape: {uded['groundTruth'].shape}")
    uded_boundaries = uded['groundTruth'][0][0]['Boundaries'][0, 0]
    print(f"UDED Boundaries shape: {uded_boundaries.shape}, dtype: {uded_boundaries.dtype}")
    
    biped = loadmat(os.path.join(output_dir, 'RGB_008.mat'))
    print(f"BIPED groundTruth shape: {biped['groundTruth'].shape}")
    biped_boundaries = biped['groundTruth'][0][0]['Boundaries'][0, 0]
    print(f"BIPED Boundaries shape: {biped_boundaries.shape}, dtype: {biped_boundaries.dtype}")
    print(f"BIPED Boundaries unique values: {np.unique(biped_boundaries)}")
