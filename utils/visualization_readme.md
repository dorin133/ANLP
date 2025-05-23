# Salient Thoughts Visualization

This directory contains scripts for visualizing salient thoughts data from Chain of Thought (CoT) reasoning analysis.

## Scripts

1. `visualize_salient_thoughts.py` - Generates individual visualizations for each example
2. `create_summary_visualizations.py` - Creates summary visualizations with all examples in one figure

## Requirements

The scripts require the following Python libraries:
- numpy
- matplotlib
- seaborn
- pandas (optional)

You can install them using conda:
```bash
conda create -n anlp_viz python=3.9 numpy matplotlib seaborn pandas -y
conda activate anlp_viz
```

## Usage

### Individual Example Visualization

To visualize a specific example:
```bash
python visualize_salient_thoughts.py --example math-algebra_0
```

To process all examples:
```bash
python visualize_salient_thoughts.py --all
```

Command line arguments:
- `--example`: Specific example to process (e.g., math-algebra_0 or aime_1)
- `--all`: Process all examples
- `--base-dir`: Base directory containing array logs (default: "ANLP/array_logs")
- `--output-dir`: Output directory for visualizations (default: "visualizations")

### Summary Visualization

To create summary visualizations for both datasets:
```bash
python create_summary_visualizations.py
```

This will generate two files:
- `visualiztions_salient_thoughts/aime/aime_ALL_examples.png`
- `visualiztions_salient_thoughts/math-algebra/math-algebra_ALL_examples.png`

## How It Works

The scripts analyze data from `salient_thoughts.npy` files, which contain attention values for different layers, heads, and thoughts in the Qwen2.5-Math model's Chain of Thought reasoning.

Structure of the data:
- Each `salient_thoughts.npy` file has shape (7, 28, 3), representing:
  - 7 layers (layers 1, 5, 9, 13, 17, 21, 25)
  - 28 attention heads per layer
  - 3 thought steps

The visualization scripts:
1. Load the NumPy arrays for each example
2. Create heatmaps showing the attention values
3. Organize the visualizations by dataset type (AIME or math-algebra)
4. For the summary visualization, arrange all examples in a grid (5 rows × 7 columns)

The visualizations help identify patterns in how different attention heads attend to different thought steps across model layers. 