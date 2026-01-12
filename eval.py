#!/usr/bin/env python3
"""
Evaluate edge detection results using ODS/OIS metrics.

Usage:
    python eval.py --results_dir results/uded_bsds --gt_dir data/UDED/gt_mat
"""
import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from scipy.io import savemat


def generate_mat_files(image_dir, output_dir=None):
    """
    Convert PNG edge detection results to MAT format for evaluation.
    """
    if output_dir is None:
        output_dir = os.path.join(image_dir, "result_mat")
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for file in os.listdir(image_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(image_dir, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read {file}")
                continue
            img = img.astype(np.float32) / 255.0
            save_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".mat")
            savemat(save_path, {"result": img})
            count += 1
    
    print(f"{count} files converted to MAT format in {output_dir}")
    return output_dir


def call_ods_ois(model, results_path, gt_mat_dir):
    """
    Run ODS/OIS evaluation using edge_eval_python's separate environment.
    """
    # Convert results to MAT format
    mat_dir = generate_mat_files(results_path)
    
    # Paths for edge_eval_python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    edge_eval_dir = os.path.join(script_dir, 'external', 'edge_eval_python')
    external_python = os.path.join(edge_eval_dir, '.venv', 'bin', 'python3')
    
    # Verify edge_eval Python exists
    if not os.path.exists(external_python):
        print(f"Error: edge_eval Python not found at {external_python}")
        print("Please set up the edge_eval_python virtual environment.")
        sys.exit(1)
    
    # Use absolute paths so subprocess cwd doesn't cause issues
    abs_mat_dir = os.path.abspath(mat_dir)
    abs_save_dir = os.path.abspath(os.path.join(mat_dir, "eval_output"))
    abs_gt_dir = os.path.abspath(gt_mat_dir)
    
    cmd = [
        external_python,
        "main.py",
        "--alg", model,
        "--model_name_list", model,
        "--result_dir", abs_mat_dir,
        "--save_dir", abs_save_dir,
        "--gt_dir", abs_gt_dir,
        "--key", "result",
        "--file_format", ".mat",
        "--workers", "1"
    ]
    
    print(f"\nRunning edge evaluation...")
    print(f"  Result dir: {abs_mat_dir}")
    print(f"  GT dir: {abs_gt_dir}")
    
    subprocess.run(cmd, cwd=edge_eval_dir)
    
    # Read and display results
    result_file = os.path.join(abs_save_dir, model + "-eval", "eval_bdry.txt")
    if os.path.exists(result_file):
        results = np.loadtxt(result_file)
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"ODS: F={results[3]:.4f} (P={results[2]:.4f}, R={results[1]:.4f})")
        print(f"OIS: F={results[6]:.4f} (P={results[5]:.4f}, R={results[4]:.4f})")
        print(f"AP:  {results[7]:.4f}")
        print("="*50)
        print(f"\nFull results saved to: {result_file}")
    
    return abs_save_dir


def main():
    parser = argparse.ArgumentParser(description="Evaluate edge detection using ODS/OIS metrics")
    parser.add_argument("--results_dir", required=True, 
                        help="Directory containing PNG edge detection results")
    parser.add_argument("--gt_dir", required=True,
                        help="Directory containing ground truth .mat files")
    parser.add_argument("--model", default="model",
                        help="Model name for output labeling (default: model)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.gt_dir):
        print(f"Error: GT directory not found: {args.gt_dir}")
        sys.exit(1)
    
    call_ods_ois(args.model, args.results_dir, args.gt_dir)


if __name__ == "__main__":
    main()
