import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def load_data_for_dataset(base_dir, dataset_type, num_examples=5):
    """Load data for all examples of a specific dataset type."""
    data_list = []
    example_names = []
    
    # Construct the glob pattern based on dataset_type
    pattern = f"Qwen2.5-Math_{dataset_type}*_*_2shot"
    
    # Find directories matching the pattern
    base_path = Path(base_dir)
    example_dirs = sorted(list(base_path.glob(pattern)))[:num_examples]
    
    for example_dir in example_dirs:
        # Extract example number
        full_name = example_dir.name.replace("Qwen2.5-Math_", "").replace("_2shot", "")
        example_num = full_name.split("_")[-1]
        example_name = f"{dataset_type}_{example_num}"
        
        # File path for salient thoughts
        salient_file = example_dir / "salient_thoughts.npy"
        
        if salient_file.exists():
            try:
                data = np.load(salient_file)
                data_list.append(data)
                example_names.append(example_name)
                print(f"Loaded data for {example_name}")
            except Exception as e:
                print(f"Error loading {salient_file}: {e}")
        else:
            print(f"File not found: {salient_file}")
    
    return data_list, example_names

def create_summary_visualization(data_list, example_names, dataset_type, output_dir):
    """Create a summary visualization with all examples for a dataset type."""
    if not data_list:
        print(f"No data available for {dataset_type}")
        return None
    
    num_examples = len(data_list)
    num_layers = data_list[0].shape[0]
    
    # Create a large figure with 5 rows (examples) and 7 columns (layers)
    fig, axes = plt.subplots(num_examples, num_layers, figsize=(24, 20))
    
    # Layer names
    layer_names = ["Layer 1", "Layer 5", "Layer 9", "Layer 13", "Layer 17", "Layer 21", "Layer 25"]
    
    # Plot each example as a row
    for row, (data, example_name) in enumerate(zip(data_list, example_names)):
        # Plot each layer as a column
        for col in range(num_layers):
            ax = axes[row, col]
            
            # Get data for this layer
            layer_data = data[col]
            
            # Plot heatmap
            sns.heatmap(layer_data, ax=ax, cmap="viridis", vmin=0, vmax=1,
                       xticklabels=[f"T{j+1}" for j in range(layer_data.shape[1])],
                       yticklabels=[f"H{j+1}" for j in range(layer_data.shape[0]) if j % 5 == 0],
                       cbar=False)
            
            # Set title and labels
            if row == 0:
                ax.set_title(layer_names[col], fontsize=12)
            
            # Add example name to leftmost plots
            if col == 0:
                ax.set_ylabel(example_name, fontsize=10, rotation=0, labelpad=40, ha='right')
            
            # Only show x-axis labels on bottom row
            if row < num_examples - 1:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Thoughts", fontsize=10)
            
            # Customize tick parameters for cleaner appearance
            ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add color bar on the right side of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Attention Value', rotation=270, labelpad=20)
    
    plt.suptitle(f"All {dataset_type.upper()} Examples - Salient Thoughts", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Create dataset-specific output directory if it doesn't exist
    dataset_output_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Save the figure
    output_path = os.path.join(dataset_output_dir, f"{dataset_type}_ALL_examples.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved summary visualization to {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return output_path

def main():
    base_dir = "ANLP/array_logs"
    output_dir = "visualizations"
    
    # Process AIME dataset
    print("Creating summary visualization for AIME dataset...")
    aime_data_list, aime_example_names = load_data_for_dataset(base_dir, "aime")
    create_summary_visualization(aime_data_list, aime_example_names, "aime", output_dir)
    
    # Process math-algebra dataset
    print("\nCreating summary visualization for math-algebra dataset...")
    math_data_list, math_example_names = load_data_for_dataset(base_dir, "math-algebra")
    create_summary_visualization(math_data_list, math_example_names, "math-algebra", output_dir)

if __name__ == "__main__":
    main() 