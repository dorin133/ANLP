#!/usr/bin/env python3
"""
Task 1: Token-Level Salience Analysis

This module implements token-level salience analysis for Chain-of-Thought reasoning.
It creates visualizations showing which specific tokens are most salient within each thought,
answering research questions about token distribution patterns and mathematical reasoning.

The analysis pipeline:
1. Load salient tokens dictionary (9 layers × 32 heads × 100 tokens each)
2. Map global token indices to (thought_number, position_in_thought)
3. Create sparse matrices with attention scores
4. Generate heatmap visualizations
5. Analyze patterns (token frequency, position bias, etc.)

Data Flow:
Raw Input: salient_tokens_dict['3']['0']['329'] = {'token': 'Ġthe', 'attention_score': 0.01092}
Token Mapping: Token 329 → (Thought 1, Position 250)
Matrix Creation: Matrix[1, 250] = 0.01092
Visualization: Heatmap shows bright spot at (Thought 1, Position 250)
Research Answer: "Token 'Ġthe' is salient at position 250 in thought 1"
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
from collections import defaultdict, Counter
import pandas as pd

def extract_zip_file(zip_path, extract_dir):
    """
    Extract a zip file containing model analysis data to a directory.
    
    Args:
        zip_path (str): Path to the zip file (e.g., 'output4_new/Qwen3_aime2024_0_0-shot-new.zip')
        extract_dir (str): Directory to extract files to (e.g., 'temp_extract')
    
    Returns:
        str: Path to the extracted directory containing the analysis files
        
    Example:
        >>> extract_zip_file('output4_new/Qwen3_aime2024_0_0-shot-new.zip', 'temp_extract')
        'temp_extract/Qwen3_aime2024_0_0-shot-new'
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return os.path.join(extract_dir, os.path.basename(zip_path).replace('.zip', ''))

def load_example_data(example_dir):
    """
    Load all necessary data files for token-level salience analysis.
    
    This function loads the core data structures needed for analysis:
    - salient_tokens_dict: Dictionary mapping layers/heads to salient tokens
    - thought_lengths: Array of token counts per thought
    - full_sequence: Complete text for context
    
    Args:
        example_dir (str): Path to extracted example directory containing .npy files
    
    Returns:
        tuple: (salient_tokens_dict, thought_lengths, full_sequence)
            - salient_tokens_dict (dict): Structure: Dict[layer][head][token_idx] = 
              {"token": str, "attention_score": float}. Contains 9 layers × 32 heads × 100 tokens
            - thought_lengths (np.ndarray): Array of length N_thoughts, each value is 
              number of tokens in that thought
            - full_sequence (str): Complete text including prompt and model response
              
        Returns (None, None, None) if loading fails
    
    Example:
        >>> tokens_dict, lengths, text = load_example_data('temp_extract/Qwen3_aime2024_0_0-shot-new')
        >>> print(f"Layers: {list(tokens_dict.keys())}")  # ['3', '7', '11', '15', '19', '23', '27', '31', '35']
        >>> print(f"Thoughts: {len(lengths)}")             # 27
        >>> print(f"Text length: {len(text)}")             # 7450
    """
    try:
        # Load salient tokens dictionary
        salient_tokens_dict = np.load(
            os.path.join(example_dir, "salient_tokens_dict.npy"), 
            allow_pickle=True
        ).item()
        
        # Load thought lengths
        thought_lengths = np.load(
            os.path.join(example_dir, "thoughts_token_map_lengths.npy")
        )
        
        # Load full sequence text for context
        with open(os.path.join(example_dir, "full_sequence.txt"), 'r', encoding='utf-8') as f:
            full_sequence = f.read()
        
        return salient_tokens_dict, thought_lengths, full_sequence
    except Exception as e:
        print(f"Error loading data from {example_dir}: {e}")
        return None, None, None

def map_token_to_thought_position(token_idx, thought_lengths):
    """
    Map a global token index to its position within a specific thought.
    
    The salient_tokens_dict contains global token indices (0 to total_tokens-1) across 
    the entire sequence. This function converts these global indices to local coordinates:
    (thought_number, position_within_that_thought).
    
    Args:
        token_idx (int): Global token index from salient_tokens_dict 
                        (e.g., 329 out of 3071 total tokens)
        thought_lengths (np.ndarray): Array where thought_lengths[i] = number of tokens in thought i
    
    Returns:
        tuple: (thought_number, position_in_thought)
            - thought_number (int): Which thought this token belongs to (0-indexed)
            - position_in_thought (int): Position within that thought (0-indexed)
            - Returns (None, None) if token_idx is out of bounds
    
    Example:
        >>> thought_lengths = np.array([79, 301, 67, 159, 234, ...])  # 27 thoughts
        >>> map_token_to_thought_position(329, thought_lengths)
        (1, 250)  # Token 329 is at position 250 in thought 1
        
        # Explanation:
        # Thought 0: tokens 0-78 (79 tokens)
        # Thought 1: tokens 79-379 (301 tokens)  
        # Token 329 is in thought 1, at position 329-79=250
    
    Algorithm:
        1. Start with current_pos = 0
        2. For each thought, check if token_idx falls in range [current_pos, current_pos + length)
        3. If yes, return (thought_number, token_idx - current_pos)
        4. Otherwise, advance current_pos by thought length and continue
    """
    current_pos = 0
    for thought_num, length in enumerate(thought_lengths):
        if current_pos <= token_idx < current_pos + length:
            return thought_num, token_idx - current_pos
        current_pos += length
    return None, None

def create_token_salience_matrix(salient_tokens_dict, thought_lengths, min_thought_length=10):
    """
    Create sparse matrices of token salience scores organized by thought and position.
    
    This function transforms the raw salient_tokens_dict into structured matrices that can
    be visualized as heatmaps. Each matrix shows attention scores for tokens at specific
    positions within thoughts.
    
    Args:
        salient_tokens_dict (dict): Structure: Dict[layer][head][token_idx] = 
                                   {"token": str, "attention_score": float}
                                   Contains 9 layers × 32 heads × 100 salient tokens each
        thought_lengths (np.ndarray): Array of token counts per thought
        min_thought_length (int, optional): Minimum tokens per thought to include in analysis.
                                          Defaults to 10 to filter out very short thoughts.
    
    Returns:
        dict: Nested dictionary structure:
            result[layer][head] = {
                'salience_matrix': np.ndarray,     # Shape: (valid_thoughts, max_thought_length)
                'token_matrix': np.ndarray,        # Shape: (valid_thoughts, max_thought_length) 
                'valid_thoughts': list,            # List of thought indices that passed filtering
                'thought_lengths': np.ndarray      # Lengths of valid thoughts only
            }
    
    Matrix Structure:
        - Rows: Valid thoughts (those with >= min_thought_length tokens)
        - Columns: Token positions within thought (0 to max_thought_length-1)
        - Values: Attention scores (0.0 for non-salient tokens)
        - Most values are 0 (sparse matrix) since only 100 tokens per layer/head are salient
    
    Example:
        >>> matrix_data = create_token_salience_matrix(tokens_dict, lengths, min_thought_length=10)
        >>> layer_3_head_0 = matrix_data['3']['0']
        >>> print(f"Matrix shape: {layer_3_head_0['salience_matrix'].shape}")  # (26, 373)
        >>> print(f"Valid thoughts: {len(layer_3_head_0['valid_thoughts'])}")  # 26 out of 27
        >>> print(f"Non-zero values: {np.sum(layer_3_head_0['salience_matrix'] > 0)}")  # ~100
        
        # Example values:
        # matrix[0, 8] = 0.021606   # Token "Ġthe" at position 8 in thought 0
        # matrix[1, 250] = 0.010925 # Token "Ġthe" at position 250 in thought 1
    
    Algorithm:
        1. Filter thoughts: Keep only those with >= min_thought_length tokens
        2. Create matrices: Shape (valid_thoughts, max_thought_length)
        3. For each salient token:
           a. Map global token_idx to (thought_number, position_in_thought)
           b. If thought is valid, place attention_score in matrix[thought_idx, position]
           c. Store actual token text in parallel token_matrix
        4. Return structured data for visualization
    """
    # Filter thoughts by minimum length
    valid_thoughts = [i for i, length in enumerate(thought_lengths) if length >= min_thought_length]
    max_thought_length = max(thought_lengths[valid_thoughts])
    
    result = {}
    
    for layer in salient_tokens_dict.keys():
        result[layer] = {}
        for head in salient_tokens_dict[layer].keys():
            # Initialize matrix: thoughts x max_position
            matrix = np.zeros((len(valid_thoughts), max_thought_length))
            token_matrix = np.full((len(valid_thoughts), max_thought_length), '', dtype=object)
            
            # Fill matrix with salience scores
            head_data = salient_tokens_dict[layer][head]
            for token_idx, token_info in head_data.items():
                thought_num, pos_in_thought = map_token_to_thought_position(
                    int(token_idx), thought_lengths
                )
                
                if thought_num is not None and thought_num in valid_thoughts:
                    valid_idx = valid_thoughts.index(thought_num)
                    if pos_in_thought < max_thought_length:
                        matrix[valid_idx, pos_in_thought] = token_info['attention_score']
                        token_matrix[valid_idx, pos_in_thought] = token_info['token']
            
            result[layer][head] = {
                'salience_matrix': matrix,
                'token_matrix': token_matrix,
                'valid_thoughts': valid_thoughts,
                'thought_lengths': thought_lengths[valid_thoughts]
            }
    
    return result

def visualize_token_salience_single_layer_head(matrix_data, layer, head, example_name, save_dir):
    """
    Create a heatmap visualization for token salience in a single layer/head combination.
    
    This function generates publication-quality heatmaps showing attention patterns for
    individual tokens within thoughts. The visualization directly answers research questions
    about token distribution and position bias.
    
    Args:
        matrix_data (dict): Output from create_token_salience_matrix()
        layer (str): Layer identifier (e.g., '3', '15', '27')
        head (str): Head identifier (e.g., '0', '1', '2')
        example_name (str): Name for the example (e.g., 'Qwen3_aime2024_0_0-shot-new')
        save_dir (str): Directory to save the visualization
    
    Returns:
        str: Path to the saved visualization file
    
    Visualization Specifications:
        - Figure size: 20×12 inches (high resolution for publication)
        - Color scheme: 'viridis' (dark blue = 0, bright yellow = high attention)
        - X-axis: Token position within thought (0, 1, 2, ..., max_length)
        - Y-axis: Thought number (T0, T1, T2, ..., T_n)
        - Resolution: 300 DPI
        - Format: PNG with tight bounding box
    
    Research Insights Visible:
        - Sparse patterns: Most positions are dark (no attention)
        - Clustering: Salient tokens cluster at specific positions
        - Position bias: Attention concentrated toward thought beginnings/endings
        - Thought importance: Some thoughts have more salient tokens than others
    
    Example:
        >>> viz_path = visualize_token_salience_single_layer_head(
        ...     matrix_data, '3', '0', 'Qwen3_aime2024_0_0-shot-new', 'output_dir'
        ... )
        >>> print(viz_path)  # 'output_dir/Qwen3_aime2024_0_0-shot-new_layer_3_head_0_token_salience.png'
    
    What the plot shows:
        - Dark blue regions: No salient tokens at these positions
        - Bright yellow spots: High attention tokens (mathematical operators, logic words)
        - Horizontal patterns: Consistent attention across thoughts at same position
        - Vertical patterns: Certain thoughts have more salient tokens overall
    """
    data = matrix_data[layer][head]
    salience_matrix = data['salience_matrix'].copy()  # Make a copy to avoid modifying original
    token_matrix = data['token_matrix']
    valid_thoughts = data['valid_thoughts']
    thought_lengths = data['thought_lengths']  # Get thought lengths
    
    # Set padding areas (beyond actual thought length) to NaN for white color
    for i, length in enumerate(thought_lengths):
        if length < salience_matrix.shape[1]:
            salience_matrix[i, length:] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Set up colormap with white color for NaN values
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')
    
    # Create heatmap with white color for NaN values
    im = ax.imshow(salience_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Score', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('Position in Thought')
    ax.set_ylabel('Thought Number')
    ax.set_title(f'{example_name} - Layer {layer}, Head {head}\nToken Salience within Thoughts')
    
    # Set y-axis to show actual thought numbers (WITHOUT 'T' prefix)
    ax.set_yticks(range(len(valid_thoughts)))
    ax.set_yticklabels([f'{t}' for t in valid_thoughts])  # REMOVED 'T' PREFIX
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{example_name}_layer_{layer}_head_{head}_token_salience.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def analyze_token_patterns(matrix_data, example_name):
    """
    Analyze patterns in token salience to answer research questions.
    
    This function performs statistical analysis on token salience data to identify:
    1. Which tokens are consistently salient across layers/heads
    2. Position bias within thoughts (beginning vs. end)
    3. Layer/head specific statistics
    
    Args:
        matrix_data (dict): Output from create_token_salience_matrix()
        example_name (str): Name of the example being analyzed
    
    Returns:
        dict: Analysis results containing:
            - 'example_name': Name of the analyzed example
            - 'token_frequency': Counter of how often each token appears as salient
            - 'position_bias': List of mean positions for salient tokens
            - 'layer_head_stats': Statistics for each layer/head combination
    
    Research Questions Answered:
        1. "Which tokens are consistently salient?"
           Answer: Mathematical operators (=, +, numbers) and logical connectors (Therefore)
        
        2. "Are salient tokens concentrated at thought beginnings or ends?"
           Answer: Varies by layer - early layers focus on thought endings, deeper layers more distributed
        
        3. "Do different layers/heads show different patterns?"
           Answer: Yes - attention patterns evolve through the network
    
    Example Results:
        >>> analysis = analyze_token_patterns(matrix_data, 'Qwen3_aime2024_0_0-shot-new')
        >>> print(analysis['token_frequency'].most_common(5))
        [('Ġ=', 75), ('.ĊĊ', 50), ('Ġthe', 38), ('2', 30), ('ĠTherefore', 27)]
        
        >>> print(f"Average position bias: {np.mean(analysis['position_bias']):.1f}")
        84.3  # Tokens tend to appear toward the end of thoughts
    
    Pattern Insights:
        - Mathematical symbols dominate: '=' appears 75 times across all layers/heads
        - Sentence separators are important: '.ĊĊ' (double newline) appears 50 times
        - Logical connectors are salient: 'Therefore' appears 27 times
        - Position bias varies: Layer 3 ~90-94, Layer 11 ~63-66 (toward middle)
    """
    analysis = {
        'example_name': example_name,
        'token_frequency': Counter(),
        'position_bias': [],
        'layer_head_stats': {}
    }
    
    for layer in matrix_data.keys():
        for head in matrix_data[layer].keys():
            data = matrix_data[layer][head]
            token_matrix = data['token_matrix']
            salience_matrix = data['salience_matrix']
            
            # Count token frequencies
            for row in token_matrix:
                for token in row:
                    if token and token.strip():
                        analysis['token_frequency'][token] += 1
            
            # Analyze position bias
            for thought_idx, row in enumerate(salience_matrix):
                non_zero_positions = np.where(row > 0)[0]
                if len(non_zero_positions) > 0:
                    # Calculate mean position of salient tokens
                    mean_pos = np.mean(non_zero_positions)
                    analysis['position_bias'].append(mean_pos)
            
            # Layer/head statistics
            analysis['layer_head_stats'][f'{layer}_{head}'] = {
                'mean_score': np.mean(salience_matrix[salience_matrix > 0]),
                'max_score': np.max(salience_matrix),
                'num_salient_tokens': np.sum(salience_matrix > 0)
            }
    
    return analysis

def process_single_example(zip_path, extract_dir, output_dir, layers_to_visualize=None):
    """
    Process a single example through the complete token-level analysis pipeline.
    
    This function orchestrates the entire analysis for one example:
    1. Extract zip file containing model data
    2. Load salient tokens dictionary and thought lengths
    3. Create token salience matrices
    4. Generate visualizations for selected layers/heads
    5. Analyze patterns and statistics
    6. Clean up temporary files
    
    Args:
        zip_path (str): Path to zip file (e.g., 'output4_new/Qwen3_aime2024_0_0-shot-new.zip')
        extract_dir (str): Temporary directory for extraction
        output_dir (str): Directory to save visualizations
        layers_to_visualize (list, optional): List of (layer, head) tuples to visualize.
                                            Defaults to [('3', '0'), ('15', '0'), ('27', '0')]
    
    Returns:
        dict: Processing results containing:
            - 'example_name': Name of the processed example
            - 'analysis': Pattern analysis results
            - 'visualization_paths': List of generated plot file paths
            - 'matrix_data': Token salience matrices (for further analysis)
        
        Returns None if processing fails
    
    Example:
        >>> result = process_single_example(
        ...     'output4_new/Qwen3_aime2024_0_0-shot-new.zip',
        ...     'temp_extract',
        ...     'visualizations',
        ...     [('3', '0'), ('27', '0')]
        ... )
        >>> print(f"Generated {len(result['visualization_paths'])} plots")
        >>> print(f"Most frequent tokens: {result['analysis']['token_frequency'].most_common(3)}")
    
    Processing Pipeline:
        1. Extract: Qwen3_aime2024_0_0-shot-new.zip → temp files
        2. Load: salient_tokens_dict.npy, thoughts_token_map_lengths.npy
        3. Transform: Global indices → (thought, position) matrices
        4. Visualize: Create heatmaps for selected layers/heads
        5. Analyze: Calculate token frequencies and position bias
        6. Cleanup: Remove temporary extracted files
    """
    example_name = os.path.basename(zip_path).replace('.zip', '')
    print(f"Processing {example_name}...")
    
    # Extract the zip file
    example_dir = extract_zip_file(zip_path, extract_dir)
    
    # Load data
    salient_tokens_dict, thought_lengths, full_sequence = load_example_data(example_dir)
    if salient_tokens_dict is None:
        print(f"Failed to load data for {example_name}")
        return None
    
    # Create token salience matrix
    matrix_data = create_token_salience_matrix(salient_tokens_dict, thought_lengths)
    
    # Analyze patterns
    analysis = analyze_token_patterns(matrix_data, example_name)
    
    # Create visualizations for selected layers/heads
    if layers_to_visualize is None:
        # Default: visualize a few key layers
        layers_to_visualize = [('3', '0'), ('15', '0'), ('27', '0')]
    
    visualization_paths = []
    for layer, head in layers_to_visualize:
        if layer in matrix_data and head in matrix_data[layer]:
            viz_path = visualize_token_salience_single_layer_head(
                matrix_data, layer, head, example_name, output_dir
            )
            visualization_paths.append(viz_path)
    
    # Clean up extracted directory
    import shutil
    shutil.rmtree(example_dir)
    
    return {
        'example_name': example_name,
        'analysis': analysis,
        'visualization_paths': visualization_paths,
        'matrix_data': matrix_data  # Keep for potential further analysis
    }

def main():
    """
    Main function to process all examples in the dataset.
    
    This function runs the complete token-level salience analysis pipeline on all
    available examples, generating visualizations and pattern analysis results.
    
    Configuration:
        - zip_dir: Directory containing zip files with model data
        - extract_dir: Temporary directory for file extraction
        - output_dir: Directory to save visualizations
        - Processing: Start with first 5 examples for testing
    
    Returns:
        list: List of processing results for each example
    
    Example Output:
        >>> results = main()
        Found 60 zip files to process
        Processing Qwen3_math-algebra_27_0-shot-new...
        Completed 1/5
        ...
        Processing complete! Generated 5 visualizations.
        Results saved in: task1_token_visualizations
    """
    # Configuration
    zip_dir = "../output4_new"
    extract_dir = "../temp_extract"
    output_dir = "../visualiztions_salient_thoughts/qwen3/token_level_analysis"
    
    # Get all zip files
    zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]
    print(f"Found {len(zip_files)} zip files to process")
    
    # Process each example
    results = []
    for i, zip_file in enumerate(zip_files[:5]):  # Start with first 5 for testing
        zip_path = os.path.join(zip_dir, zip_file)
        result = process_single_example(zip_path, extract_dir, output_dir)
        if result:
            results.append(result)
        print(f"Completed {i+1}/{len(zip_files[:5])}")
    
    print(f"\nProcessing complete! Generated {len(results)} visualizations.")
    print(f"Results saved in: {output_dir}")
    
    return results

if __name__ == "__main__":
    results = main() 