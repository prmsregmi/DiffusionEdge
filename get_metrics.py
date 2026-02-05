import os
import glob
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Get metrics for a specific model.')
    parser.add_argument('--model_name', type=str, help='Name of the model to filter results by (suffix of directory name)')
    args = parser.parse_args()

    # Define the pattern to search for metrics.csv files
    # The structure described is: results/{FOLDER_NAME}/result_mat/eval_output/{MODEL_NAME}/metrics.csv
    # We use a glob pattern to match this structure
    if args.model_name:
        search_pattern = f"*_{args.model_name}"
    else:
        search_pattern = "*"
        
    search_path = os.path.join("results", search_pattern, "result_mat", "eval_output", "*", "metrics.csv")
    
    files = glob.glob(search_path)
    files.sort()
    
    if not files:
        print(f"No metrics files found in {search_path}", file=sys.stderr)
        return

    # Print Header removed as we are printing per-line labels

    for file_path in files:
        try:
            # Extract the folder name (Experiment name)
            # path is like: results/BIPED_biped/result_mat/...
            # splitting by os.sep will give us components
            parts = file_path.split(os.sep)
            
            # parts[0] is 'results', parts[1] is the experiment folder
            experiment_name = parts[1]
            
            with open(file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                
                # Check if file has content
                if len(lines) > 1:
                    # The first line is the header: ODS,OIS,AP,R50
                    # The second line contains the values
                    try:
                        vals = lines[1].strip().split(',')
                        if len(vals) >= 3:
                            ods = float(vals[0])
                            ois = float(vals[1])
                            ap = float(vals[2])
                            
                            # Format: ODS=0.693, OIS=0.694, AP=0.533
                            print(f"{experiment_name}: ODS={ods:.3f}, OIS={ois:.3f}, AP={ap:.3f}")
                        else:
                             print(f"{experiment_name}: Malformed data", file=sys.stderr)
                    except ValueError:
                        print(f"{experiment_name}: Error parsing values", file=sys.stderr)
                else:
                    # Handle empty or header-only files
                    print(f"{experiment_name}: No Data", file=sys.stderr)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
