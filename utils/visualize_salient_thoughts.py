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


# def visualize_salient_thoughts(data, title, output_dir):
#     """Create heatmap visualizations for a single example."""
#     # Check data shape
#     num_layers, num_heads, num_thoughts = data.shape
#
#     # Calculate number of rows needed (2 plots per row)
#     num_rows = (num_layers + 1) // 2  # Using integer division and ceiling
#
#     # Create a figure with subplots arranged in multiple rows, 2 columns
#     fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), sharey=False)
#
#     # Layer names (based on project description)
#     layer_names = ["Layer 3", "Layer 7", "Layer 11", "Layer 15", "Layer 19", "Layer 23", "Layer 27"]
#
#     # Create heatmaps for each layer
#     for i in range(num_layers):
#         # Calculate row and column position
#         row = i // 2
#         col = i % 2
#
#         ax = axes[row, col]
#
#         # Get data for this layer
#         layer_data = data[i]
#
#         # Plot heatmap
#         sns.heatmap(layer_data, ax=ax, cmap="viridis", vmin=0, vmax=1,
#                     xticklabels=[f"{j + 1}" for j in range(num_thoughts)],
#                     yticklabels=[f"{j + 1}" for j in range(num_heads)])
#
#         # Set title and labels
#         ax.set_title(layer_names[i])
#         ax.set_ylabel("Attention Heads")
#         ax.set_xlabel("Thoughts")
#
#     # Hide any unused subplots
#     for i in range(num_layers, num_rows * 2):
#         row = i // 2
#         col = i % 2
#         fig.delaxes(axes[row, col])
#
#     plt.suptitle(title, fontsize=16, y=1.01)
#     plt.tight_layout()
#
#     # Save the figure
#     output_path = os.path.join(output_dir, f"salient_thoughts.png")
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     print(f"Saved visualization to {output_path}")
#
#     # Close the figure to free memory
#     plt.close(fig)
#
#     return output_path


def visualize_salient_thoughts(data, title, output_dir):
    """Wrapper that prepares path and delegates to core heatmap function."""
    output_path = os.path.join(output_dir, "salient_thoughts.png")
    return visualize_thoughts_analysis(
        data=data,
        title=title,
        output_dir=output_path,
        xlabel="Thoughts",
        ylabel="Attention Heads",
        vmin=0,
        vmax=1)

def visualize_thoughts_interactions(data, title, output_path):
    """Wrapper that prepares path and delegates to core heatmap function."""
    return visualize_thoughts_analysis(
        data=data,
        title=title,
        output_dir=output_path,
        xlabel="Previous Thoughts Index",
        ylabel="Current Thoughts Index",
        vmin=0,
        vmax=None)


def visualize_thoughts_analysis(data, title, output_dir, xlabel="Thoughts", ylabel="Attention Heads", vmin=None, vmax=None):
    """
    Create heatmap visualizations for a single example.

    Parameters:
        data (np.ndarray): 3D array of shape (num_layers, num_heads, num_thoughts)
        title (str): Title for the entire figure
        output_dir (str): Directory to save the output image
        xlabel (str): Label for x-axis (default: "Thoughts")
        ylabel (str): Label for y-axis (default: "Attention Heads")
    """
    # Check data shape
    num_layers, num_heads, num_thoughts = data.shape

    # Calculate number of rows needed (2 plots per row)
    num_rows = (num_layers + 1) // 2  # ceiling division

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows), sharey=False)

    # Make axes 2D even if there's only one row
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # Layer names (can be extended or passed as param)
    layer_names = [f"Layer {i * 4 + 3}" for i in range(num_layers)]

    # Create heatmaps for each layer
    for i in range(num_layers):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        layer_data = data[i]

        sns.heatmap(layer_data, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                    xticklabels=[f"{j + 1}" for j in range(num_thoughts)],
                    yticklabels=[f"{j + 1}" for j in range(num_heads)])

        ax.set_title(layer_names[i])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide any unused subplots
    for i in range(num_layers, num_rows * 2):
        row = i // 2
        col = i % 2
        fig.delaxes(axes[row, col])

    plt.suptitle(title, fontsize=16, y=0.098)
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, f"salient_thoughts.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved visualization to {output_path}")
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
                visualize_salient_thoughts(data, example_name, "aime", output_dir)
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
                visualize_salient_thoughts(data, example_name, "math-algebra", output_dir)
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
        dataset_name = "aime" if "aime" in args.example else "math-algebra"
        example_dir = os.path.join(args.base_dir, f"Qwen2.5-Math_{args.example}_2shot")
        salient_file = os.path.join(example_dir, "salient_thoughts.npy")
        
        if os.path.exists(salient_file):
            data = load_salient_thoughts(salient_file)
            if data is not None:
                visualize_salient_thoughts(data, args.example, dataset_name, args.output_dir)
        else:
            print(f"File not found: {salient_file}")
    
    elif args.all:
        # Process all examples
        process_all_examples(args.base_dir, args.output_dir)
    
    else:
        print("Please specify either --example or --all")

if __name__ == "__main__":
    main() 