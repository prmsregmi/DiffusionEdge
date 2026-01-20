#!/usr/bin/env python3
import argparse
import os
import subprocess
import json
import re
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run DiffusionEdge 3x4 experiments")
    parser.add_argument("--models", type=str, default="bsds,nyud,biped",
                        help="Comma-separated list of models to evaluate (default: bsds,nyud,biped)")
    parser.add_argument("--test", type=str, default="BIPED,UDED,BSDS,NYUD",
                        help="Comma-separated list of test datasets (default: BIPED,UDED,BSDS,NYUD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    return parser.parse_args()

def check_dataset_exists(dataset_name):
    # Check for both imgs and gt_mat
    imgs_path = os.path.join("data", dataset_name, "imgs")
    gt_path = os.path.join("data", dataset_name, "gt_mat")
    
    # We might need to handle case sensitivity or flexible paths, 
    # but for now strict check as per user request context
    if not os.path.isdir(imgs_path):
        print(f"Warning: Image directory not found: {imgs_path}")
        return False
    if not os.path.isdir(gt_path):
        print(f"Warning: GT directory not found: {gt_path}")
        return False
    return True

def parse_eval_output(output):
    """
    Parse ODS and OIS scores from eval.py output.
    Expected format:
    ODS: F=0.xxxx ...
    OIS: F=0.xxxx ...
    """
    ods_f = 0.0
    ois_f = 0.0
    
    ods_match = re.search(r"ODS: F=([0-9.]+)", output)
    if ods_match:
        ods_f = float(ods_match.group(1))
        
    ois_match = re.search(r"OIS: F=([0-9.]+)", output)
    if ois_match:
        ois_f = float(ois_match.group(1))
        
    return ods_f, ois_f

def run_command(cmd, dry_run=False, capture_output=True):
    print(f"Running: {' '.join(cmd)}")
    if dry_run:
        return "ODS: F=0.0 (P=0.0, R=0.0)\nOIS: F=0.0 (P=0.0, R=0.0)"
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            return result.stdout
        else:
            # Stream directly to console
            subprocess.run(cmd, check=True)
            return "SUCCESS"
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture_output:
            print(e.stdout)
            print(e.stderr, file=sys.stderr)
        return None

def main():
    args = parse_args()
    
    train_models = [m.strip() for m in args.models.split(",")]
    test_datasets = [d.strip() for d in args.test.split(",")]
    
    results = {}
    
    for train_model in train_models:
        for test_dataset in test_datasets:
            print(f"\n{'='*50}")
            print(f"Experiment: Model={train_model}, Test={test_dataset}")
            print(f"{'='*50}")
            
            if not check_dataset_exists(test_dataset):
                print(f"Skipping {test_dataset} (missing files)")
                continue
                
            out_dir = os.path.join("results", f"{test_dataset}_{train_model}")
            input_dir = os.path.join("data", test_dataset, "imgs")
            gt_dir = os.path.join("data", test_dataset, "gt_mat")
            
            # 1. Run Demo (Inference)
            demo_cmd = [
                sys.executable, "demo.py",
                "--model", train_model,
                "--input_dir", input_dir,
                "--out_dir", out_dir,
                "--skip_small",
                "--sampling_timesteps", "5"
            ]
            
            # Don't capture output for demo, so user sees progress
            demo_output = run_command(demo_cmd, args.dry_run, capture_output=False)
            if demo_output is None and not args.dry_run:
                print("Inference failed, skipping evaluation.")
                continue

            # 2. Run Eval (Evaluation)
            eval_cmd = [
                sys.executable, "eval.py",
                "--results_dir", out_dir,
                "--gt_dir", gt_dir,
                "--model", train_model
            ]
            
            # Capture output for eval to parse scores
            eval_output = run_command(eval_cmd, args.dry_run, capture_output=True)
            if eval_output:
                ods, ois = parse_eval_output(eval_output)
                
                key = f"{train_model}/{test_dataset}"
                results[key] = {
                    "train_model": train_model,
                    "test_dataset": test_dataset,
                    "ODS": ods,
                    "OIS": ois
                }
                print(f"Result: {key} -> ODS={ods}, OIS={ois}")

    # Save to JSON
    output_file = "final_results.json"
    if args.dry_run:
        print(f"[Dry Run] Would save results to {output_file}")
        print(json.dumps(results, indent=4))
    else:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nSaved final results to {output_file}")

if __name__ == "__main__":
    main()
