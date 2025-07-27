# Salient Thoughts Visualization

This directory contains scripts for visualizing salient thoughts and token-level salience data from Chain of Thought (CoT) reasoning analysis on both Qwen2.5 and Qwen3 models.

## Scripts

### Core Visualization Functions
1. `visualize_salient_thoughts.py` - Core functions for generating thought-level heatmap visualizations
2. `create_summary_visualizations.py` - Creates summary visualizations with all examples in one figure

### Qwen3 Analysis Scripts
3. `../auxiliaries/token_level_salience_analysis.py` - Complete pipeline for token-level salience analysis
4. `../auxiliaries/thought_level_salience_analysis.py` - Batch processing for thought-level analysis on 60 examples
5. `../scripts/run_token_level_analysis.sh` - Shell script for automated token-level analysis
6. `../scripts/run_thought_level_analysis.sh` - Shell script for automated thought-level analysis

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

### Qwen3 Analysis (Recommended - Latest)

#### Token-Level Salience Analysis
Analyze which specific tokens are most important within each thought:

```bash
# Run from project root
cd anlp_v2
./scripts/run_token_level_analysis.sh
```

This generates heatmaps showing salient tokens at specific positions within thoughts. Results saved to:
- `visualiztions_salient_thoughts/qwen3/token_level_analysis/`

#### Thought-Level Salience Analysis  
Analyze which entire thoughts are most important across layers and heads:

```bash
# Run from project root
cd anlp_v2
./scripts/run_thought_level_analysis.sh
```

Processes all 60 examples (30 AIME + 30 math-algebra) with 9 layers × 32 heads. Results saved to:
- `visualiztions_salient_thoughts/qwen3/thought_level_analysis/`

#### Manual Python Execution
```bash
# Token-level analysis
cd anlp_v2/auxiliaries
python token_level_salience_analysis.py

# Thought-level analysis
cd anlp_v2/auxiliaries
python thought_level_salience_analysis.py
```

### Qwen2.5 Analysis (Legacy)

#### Individual Example Visualization

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

#### Summary Visualization

To create summary visualizations for both datasets:
```bash
python create_summary_visualizations.py
```

Legacy results are preserved in:
- `visualiztions_salient_thoughts/qwen2.5/`

## How It Works

### Qwen3 Analysis Architecture

#### Token-Level Salience Analysis
Analyzes which specific tokens within thoughts are most important for the model's reasoning:

**Data Sources:**
- `salient_tokens_dict.npy`: Dictionary structure `Dict[layer][head][token_idx] = {"token": str, "attention_score": float}`
- `thoughts_token_map_lengths.npy`: Array of token counts per thought for filtering and normalization

**Analysis Pipeline:**
1. **Data Loading**: Extract zip files and load salient tokens dictionary (9 layers × 32 heads × 100 tokens each)
2. **Token Mapping**: Convert global token indices to (thought_number, position_in_thought) coordinates
3. **Matrix Creation**: Build sparse matrices showing attention scores for tokens at specific positions within thoughts
4. **Filtering**: Only analyze thoughts with ≥10 tokens to focus on substantial reasoning steps
5. **Visualization**: Generate heatmaps where X=token_position_in_thought, Y=thought_number, Color=attention_score
6. **Pattern Analysis**: Identify which tokens are consistently salient (mathematical operators, logical connectors)

**Key Insights:**
- Mathematical symbols (`=`, `+`, numbers) are consistently salient across examples
- Logical connectors (`Therefore`, sentence separators) show high attention
- Position bias varies by layer: early layers focus on thought endings, deeper layers more distributed

#### Thought-Level Salience Analysis
Analyzes which entire reasoning steps (thoughts) are most important:

**Data Sources:**
- `salient_thoughts.npy`: 3D array shape (9_layers, 32_heads, N_thoughts) containing salience scores for each thought

**Analysis Pipeline:**
1. **Batch Processing**: Stream through 60 zip files (30 AIME + 30 math-algebra)
2. **Visualization**: Create 9-layer heatmaps (layers 3, 7, 11, 15, 19, 23, 27, 31, 35) showing thought importance
3. **Memory Management**: Process one example at a time to handle large datasets efficiently

**Architecture Differences:**
- **Qwen3**: 9 layers × 32 heads, 0-shot prompting, 60 examples
- **Qwen2.5**: 7 layers × 28 heads, 2-shot prompting, 20 examples

### Qwen2.5 Analysis (Legacy)

The original scripts analyze data from `salient_thoughts.npy` files with:
- Shape (7, 28, 3) representing 7 layers, 28 attention heads, 3 thought steps
- Layers 1, 5, 9, 13, 17, 21, 25

### Directory Structure

```
visualiztions_salient_thoughts/
├── qwen2.5/           # Legacy Qwen2.5 analysis results
│   ├── *.png          # Individual and summary visualizations
└── qwen3/             # New Qwen3 analysis results  
    ├── token_level_analysis/     # Token-level heatmaps
    │   └── *.png
    └── thought_level_analysis/   # Thought-level heatmaps
        ├── aime_*/salient_thoughts.png
        └── math-algebra_*/salient_thoughts.png
```

The visualizations help identify patterns in how different attention heads attend to different thought steps and specific tokens across model layers, enabling deeper understanding of Chain-of-Thought reasoning mechanisms.

## Output Examples

### Token-Level Heatmaps
- **X-axis**: Token position within thought (0 to max_thought_length)
- **Y-axis**: Thought number (filtered to thoughts with ≥10 tokens)
- **Color**: Attention score intensity
- **White regions**: Beyond actual thought length (padding)

### Thought-Level Heatmaps
- **X-axis**: Thought number (1 to N, varies per example)
- **Y-axis**: Attention head (1 to 32)
- **Subplots**: 9 layers arranged in grid format
- **Color**: Normalized attention scores (0-1 range) 
