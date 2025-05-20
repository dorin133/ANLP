import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path

def load_salient_thoughts(file_path):
    """Load salient thoughts data from a .npy file."""
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def visualize_example(data, example_name, dataset_type, output_dir):
    """Create heatmap visualizations for a single example."""
    # Check data shape
    num_layers, num_heads, num_thoughts = data.shape
    
    # Create a figure with subplots for each layer
    fig, axes = plt.subplots(1, num_layers, figsize=(20, 5), sharey=True)
    
    # Layer names (based on project description)
    layer_names = ["Layer 1", "Layer 5", "Layer 9", "Layer 13", "Layer 17", "Layer 21", "Layer 25"]
    
    # Create heatmaps for each layer
    for i in range(num_layers):
        ax = axes[i]
        
        # Get data for this layer
        layer_data = data[i]
        
        # Plot heatmap
        sns.heatmap(layer_data, ax=ax, cmap="viridis", vmin=0, vmax=1,
                   xticklabels=[f"Thought {j+1}" for j in range(num_thoughts)],
                   yticklabels=[f"Head {j+1}" for j in range(num_heads)])
        
        # Set title and labels
        ax.set_title(layer_names[i])
        if i == 0:
            ax.set_ylabel("Attention Heads")
        ax.set_xlabel("Thoughts")
    
    plt.suptitle(f"Salient Thoughts - {example_name}", fontsize=16)
    plt.tight_layout()
    
    # Create dataset-specific output directory if it doesn't exist
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(dataset_output_dir, f"{example_name}_salient_thoughts.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

def process_all_examples(base_dir, output_dir="visualizations"):
    """Process all examples in the array_logs directory, organized by dataset type."""
    base_path = Path(base_dir)
    
    # Track counts by dataset type
    aime_count = 0
    math_algebra_count = 0
    
    # Find all directories that match the patterns for each dataset type
    aime_dirs = list(base_path.glob("Qwen2.5-Math_aime*_*_2shot"))
    math_algebra_dirs = list(base_path.glob("Qwen2.5-Math_math-algebra*_*_2shot"))
    
    print(f"Found {len(aime_dirs)} AIME examples and {len(math_algebra_dirs)} math-algebra examples")
    
    # Process AIME examples (limit to 5)
    for example_dir in sorted(aime_dirs)[:5]:
        # Get example name from directory name
        full_name = example_dir.name.replace("Qwen2.5-Math_", "").replace("_2shot", "")
        example_num = full_name.split("_")[-1]
        example_name = f"aime_{example_num}"
        
        # File path for salient thoughts
        salient_file = example_dir / "salient_thoughts.npy"
        
        if salient_file.exists():
            print(f"Processing AIME example: {example_name}")
            data = load_salient_thoughts(salient_file)
            
            if data is not None:
                visualize_example(data, example_name, "aime", output_dir)
                aime_count += 1
        else:
            print(f"No salient_thoughts.npy found in {example_dir}")
    
    # Process math-algebra examples (limit to 5)
    for example_dir in sorted(math_algebra_dirs)[:5]:
        # Get example name from directory name
        full_name = example_dir.name.replace("Qwen2.5-Math_", "").replace("_2shot", "")
        example_num = full_name.split("_")[-1]
        example_name = f"math_algebra_{example_num}"
        
        # File path for salient thoughts
        salient_file = example_dir / "salient_thoughts.npy"
        
        if salient_file.exists():
            print(f"Processing math-algebra example: {example_name}")
            data = load_salient_thoughts(salient_file)
            
            if data is not None:
                visualize_example(data, example_name, "math-algebra", output_dir)
                math_algebra_count += 1
        else:
            print(f"No salient_thoughts.npy found in {example_dir}")
    
    print(f"Processed {aime_count} AIME examples and {math_algebra_count} math-algebra examples")

def main():
    parser = argparse.ArgumentParser(description="Visualize salient thoughts data")
    parser.add_argument("--example", type=str, help="Specific example to process (e.g., math-algebra_0)")
    parser.add_argument("--all", action="store_true", help="Process all examples")
    parser.add_argument("--base-dir", type=str, default="ANLP/array_logs", 
                        help="Base directory containing array logs")
    parser.add_argument("--output-dir", type=str, default="visualizations", 
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    if args.example:
        # Process a single example
        # Determine dataset type from example name
        dataset_type = "aime" if "aime" in args.example else "math-algebra"
        example_dir = os.path.join(args.base_dir, f"Qwen2.5-Math_{args.example}_2shot")
        salient_file = os.path.join(example_dir, "salient_thoughts.npy")
        
        if os.path.exists(salient_file):
            data = load_salient_thoughts(salient_file)
            if data is not None:
                visualize_example(data, args.example, dataset_type, args.output_dir)
        else:
            print(f"File not found: {salient_file}")
    
    elif args.all:
        # Process all examples
        process_all_examples(args.base_dir, args.output_dir)
    
    else:
        print("Please specify either --example or --all")

if __name__ == "__main__":
    main() 