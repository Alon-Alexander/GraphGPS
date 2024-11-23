import matplotlib.pyplot as plt
import numpy as np
import json
import os


import json

def convert_to_json(file_path, output_path):
    """
    Converts a file with multiple JSON objects (one per line) into a single JSON array.
    
    Args:
        file_path (str): Path to the input file containing JSON objects, one per line.
        output_path (str): Path to save the converted JSON file as a valid JSON array.
    """
    try:
        # Read the file line by line
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Parse each line as a JSON object
        json_objects = [json.loads(line.strip()) for line in lines if line.strip()]
        
        # Write the list of JSON objects to a new file as a JSON array
        with open(output_path, 'w') as output_file:
            json.dump(json_objects, output_file, indent=4)
        
        print(f"Successfully converted to JSON and saved to {output_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

import json
import matplotlib.pyplot as plt

def analyze_mae(json_file_path):
    """
    Analyzes the MAE (Mean Absolute Error) over epochs from a JSON file
    and visualizes it using matplotlib.
    
    Args:
        json_file_path (str): Path to the JSON file containing run data.
    """
    try:
        # Load the JSON data
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        # Extract epochs and MAE values
        epochs = [entry['epoch'] for entry in data]
        mae_values = [entry['mae'] for entry in data]
        
        # Plot the MAE values over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mae_values, marker='o', linestyle='-', label='MAE')
        plt.title('Mean Absolute Error (MAE) Over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except KeyError as e:
        print(f"Missing expected key in JSON data: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

import json
import matplotlib.pyplot as plt

def plot_two_runs(run1_path, run2_path, labels, title="MAE Comparison"):
        """
        Plots MAE for two runs on the same graph, truncated at the shorter run.
        
        Args:
            run1_path (str): Path to the first JSON file containing run data.
            run2_path (str): Path to the second JSON file containing run data.
            labels (list of str): List of labels for the two runs, e.g., ["Run 1", "Run 2"].
            title (str): Title for the plot.
        """
        try:
            # Load the data for the first run
            with open(run1_path, 'r') as file1:
                run1_data = json.load(file1)
            epochs1 = [entry['epoch'] for entry in run1_data]
            mae1 = [entry['mae'] for entry in run1_data]
            
            # Load the data for the second run
            with open(run2_path, 'r') as file2:
                run2_data = json.load(file2)
            epochs2 = [entry['epoch'] for entry in run2_data]
            mae2 = [entry['mae'] for entry in run2_data]
            
            # Truncate the data to the shorter run
            max_epochs = min(len(epochs1), len(epochs2))
            epochs1 = epochs1[:max_epochs]
            mae1 = mae1[:max_epochs]
            epochs2 = epochs2[:max_epochs]
            mae2 = mae2[:max_epochs]
            
            # Plot both runs
            plt.figure(figsize=(10, 6))
            plt.plot(epochs1, mae1, linestyle='--', label=f'{labels[0]}')
            plt.plot(epochs2, mae2, linestyle='-', label=f'{labels[1]}')
            
            # Add labels, title, legend, and grid
            plt.title(title, fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
            plt.grid(True)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()
        
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except KeyError as e:
            print(f"Missing expected key in JSON data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
# Example usage
'''
plot_two_runs(
    'stats_converted.json', 
    'decider/test_stats_converted.json', 
    labels=["Original", "Ours"], 
    title="ZINC MAE Comparison"
)
'''

# Example usage
#convert_to_json('./decider/test_stats.json', './decider/test_stats_converted.json')
# val
#convert_to_json('./decider/val_stats.json', './decider/val_stats_converted.json')

#analyze_mae('./decider/test_stats_converted.json')
#analyze_mae('./decider/val_stats_converted.json')

def attn_use(p):
    with open(p, 'r') as f:
        lines = f.readlines()
        use = [float(line.split(' ')[3].strip(',')) for line in lines]
    x = np.arange(len(use))
    plt.plot(x, use)
    #attn_use against time
    plt.xlabel('Time')
    plt.ylabel('Attention Penalty Rate')
    plt.title('Attention Usage Over Time')
    plt.show()

attn_use('analisys.txt')