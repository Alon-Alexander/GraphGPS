import re
import json

def extract_data(split, path, out):
    """
    Extracts data from a structured log file, aggregates and processes it,
    and writes the data to a JSON file where each line is a standalone JSON object.
    
    Args:
        split (str): The specific data split to extract ('train', 'val', or 'test').
        path (str): The path to the input log file.
        out (str): The path to the output JSON file.
    """
    # Adjusted regular expression to match both full and partial log entries
    pattern = re.compile(
        f"{split}: {{'epoch': (\\d+), 'time_epoch': ([\\d.]+)"
        f"(?:, 'eta': ([\\d.]+), 'eta_hours': ([\\d.]+))?"  # Optional eta and eta_hours
        f", 'loss': ([\\d.]+), 'lr': ([\\d.e-]+), 'params': (\\d+), 'time_iter': ([\\d.]+)"
        f", 'mae': ([\\d.]+), 'r2': ([\\d.-]+), 'spearmanr': ([\\d.]+), 'mse': ([\\d.]+), 'rmse': ([\\d.]+)}}"
    )
    
    with open(path, 'r') as file, open(out, 'w') as json_file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract matched groups and handle optional fields
                data_entry = {
                    "epoch": int(match.group(1)),
                    "time_epoch": float(match.group(2)),
                    "loss": float(match.group(5)),
                    "lr": float(match.group(6)),
                    "params": int(match.group(7)),
                    "time_iter": float(match.group(8)),
                    "mae": float(match.group(9)),
                    "r2": float(match.group(10)),
                    "spearmanr": float(match.group(11)),
                    "mse": float(match.group(12)),
                    "rmse": float(match.group(13))
                }
                
                # Add optional fields if present
                if match.group(3) and match.group(4):
                    data_entry["eta"] = float(match.group(3))
                    data_entry["eta_hours"] = float(match.group(4))
                
                # Write each entry as a standalone JSON line
                json_file.write(json.dumps(data_entry) + "\n")
# Example usage
extract_data("train", "run_composition_2_resume.out", "train.json")
extract_data("val", "run_composition_2_resume.out", "val.json")
extract_data("test", "run_composition_2_resume.out", "test.json")