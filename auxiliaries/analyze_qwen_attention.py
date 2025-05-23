import numpy as np
import matplotlib.pyplot as plt
import torch
import gzip
import io
from analp_functions_for_analysis import (
    identify_assistant_tokens,
    calculate_thought_interaction,
    identify_salient_thoughts
)
from auxiliaries.test_qwen_analysis import simple_tokenize, identify_thoughts, get_thought_text

def load_compressed_pytorch(filepath):
    """Load a compressed PyTorch file (.pt.gz)"""
    with gzip.open(filepath, 'rb') as f:
        buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=torch.device('cpu'))

# Load the compressed attention matrix file
print("Loading compressed PyTorch attention data...")
attention_data = load_compressed_pytorch('Qwen2.5-Math_math-algebra_2_2shot.pt.gz')

# Extract the first attention matrix
print("Extracting first attention matrix...")
print(f"Data type: {type(attention_data)}")

# Check what we've got
if isinstance(attention_data, tuple):
    print(f"Tuple length: {len(attention_data)}")
    for i, item in enumerate(attention_data):
        print(f"Item {i} type: {type(item)}")
        if isinstance(item, torch.Tensor):
            print(f"  Shape: {item.shape}")
        elif isinstance(item, (list, tuple)):
            print(f"  Length: {len(item)}")
            if len(item) > 0:
                print(f"  First element type: {type(item[0])}")
                if isinstance(item[0], torch.Tensor):
                    print(f"    Shape: {item[0].shape}")

# Try to find the attention matrix
first_attention_matrix = None
try:
    if isinstance(attention_data, tuple):
        # Try the first item if it's a tensor and has a shape that might be an attention matrix
        if len(attention_data) > 0 and isinstance(attention_data[0], torch.Tensor):
            tensor = attention_data[0]
            if len(tensor.shape) == 2 and tensor.shape[0] > 0 and tensor.shape[1] > 0:
                first_attention_matrix = tensor
                print(f"Using first tensor with shape {tensor.shape}")
        
        # If not found in first position, look through all items in tuple
        if first_attention_matrix is None:
            for i, item in enumerate(attention_data):
                if isinstance(item, torch.Tensor) and len(item.shape) == 2:
                    first_attention_matrix = item
                    print(f"Found tensor at position {i} with shape {item.shape}")
                    break
                elif isinstance(item, (list, tuple)) and len(item) > 0:
                    # Try the first element of nested lists/tuples
                    if isinstance(item[0], torch.Tensor) and len(item[0].shape) == 2:
                        first_attention_matrix = item[0]
                        print(f"Found tensor in nested list at position {i}, shape {item[0].shape}")
                        break
except Exception as e:
    print(f"Error accessing attention matrix: {e}")

if first_attention_matrix is None:
    print("Could not find the attention matrix. Please print more information about the data structure.")
    if isinstance(attention_data, tuple) and len(attention_data) > 0:
        # Try to get the first tensor we find of any shape
        for i, item in enumerate(attention_data):
            if isinstance(item, torch.Tensor):
                print(f"Found tensor at position {i} with shape {item.shape}")
                first_attention_matrix = item
                break
            elif isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], torch.Tensor):
                print(f"Found tensor in list at position {i} with shape {item[0].shape}")
                first_attention_matrix = item[0]
                break

if first_attention_matrix is None:
    print("Still could not find any tensor. Exiting.")
    exit(1)

# Convert to numpy for our functions
print("Converting tensor to numpy array...")
# Convert from BFloat16 to Float32 first
first_attention_matrix = first_attention_matrix.to(torch.float32)
attention_matrix = first_attention_matrix.numpy()
print(f"Extracted attention matrix with shape: {attention_matrix.shape}")

# Check if it's a square matrix as expected for attention
if len(attention_matrix.shape) != 2 or attention_matrix.shape[0] != attention_matrix.shape[1]:
    print(f"Warning: The matrix is not square ({attention_matrix.shape}). This may not be an attention matrix.")
    
    # If it's a 3D tensor, it might be multiple attention matrices or heads
    if len(attention_matrix.shape) == 3:
        print("Found a 3D tensor, using the first slice as attention matrix")
        attention_matrix = attention_matrix[0]
    # If it's a 4D tensor, it might be [layers, heads, seq_len, seq_len]
    elif len(attention_matrix.shape) == 4:
        print("Found a 4D tensor, using the first layer and head as attention matrix")
        attention_matrix = attention_matrix[0, 0]

print(f"Final attention matrix shape: {attention_matrix.shape}")

# Load and process the text file
with open('Qwen2.5-Math_math-algebra_2_2shot_full_sequence.txt', 'r') as f:
    content = f.read()

# Basic tokenization (this is a simplified approach)
tokens = simple_tokenize(content)
print(f"Total tokens from text: {len(tokens)}")

# Check if matrix dimensions match token count
if attention_matrix.shape[0] != len(tokens) or attention_matrix.shape[1] != len(tokens):
    print(f"Warning: Matrix dimensions ({attention_matrix.shape}) don't match token count ({len(tokens)}).")
    print("Using the smaller dimension for analysis.")
    token_count = min(len(tokens), attention_matrix.shape[0], attention_matrix.shape[1])
    tokens = tokens[:token_count]
    attention_matrix = attention_matrix[:token_count, :token_count]

# Make sure the attention matrix is normalized (rows sum to 1)
row_sums = attention_matrix.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1.0, row_sums)  # Avoid division by zero
attention_matrix = attention_matrix / row_sums
print(f"Normalized attention matrix with shape: {attention_matrix.shape}")

# Identify assistant tokens
assistant_mask = identify_assistant_tokens(tokens)
assistant_start_idx = assistant_mask.index(True) if True in assistant_mask else -1
print(f"Assistant tokens start at index: {assistant_start_idx}")
print(f"Number of assistant tokens: {sum(assistant_mask)}")

# Group tokens into thoughts
thoughts_token_map = identify_thoughts(tokens)
print(f"Number of thoughts identified: {len(thoughts_token_map)}")

# Identify which thoughts contain assistant tokens
assistant_thoughts = []
for i, thought_indices in enumerate(thoughts_token_map):
    if any(assistant_mask[idx] for idx in thought_indices):
        assistant_thoughts.append(i)
print(f"Assistant thoughts: {assistant_thoughts}")

# Print examples of thoughts
print("\n--- Example Thought Segments ---")
num_examples = min(3, len(thoughts_token_map))
for i in range(num_examples):
    print(f"Thought {i}: {get_thought_text(tokens, thoughts_token_map[i])}")

print("\n--- Example Assistant Thought Segments ---")
for i in assistant_thoughts[:3]:  # First few assistant thoughts
    print(f"Thought {i}: {get_thought_text(tokens, thoughts_token_map[i])}")

# Analyze thought interactions but focus only on assistant thoughts
all_interactions = {}

print("\n--- Thought Interaction Analysis (Assistant Thoughts Only) ---")
for i in assistant_thoughts:
    interactions = calculate_thought_interaction(
        attention_matrix,
        thoughts_token_map,
        i,
        K=3,
        assistant_mask=assistant_mask
    )
    
    if interactions:
        all_interactions[i] = interactions
        
        # Find the most significant previous thought
        most_significant = None
        highest_score = 0
        for prev_idx, scores in interactions.items():
            if scores['mean_all_scores'] > highest_score:
                highest_score = scores['mean_all_scores']
                most_significant = prev_idx
        
        if most_significant is not None and highest_score > 0.01:  # Only show significant interactions
            print(f"\nThought {i} most attends to Thought {most_significant} (score: {highest_score:.4f}):")
            current_text = get_thought_text(tokens, thoughts_token_map[i])
            prev_text = get_thought_text(tokens, thoughts_token_map[most_significant])
            print(f"  Current: {current_text}")
            print(f"  Attends to: {prev_text}")

# Calculate token salience (column sum of the attention matrix)
token_salience_scores = attention_matrix.sum(axis=0)

# Identify salient thoughts
salient_histogram = identify_salient_thoughts(
    token_salience_scores,
    thoughts_token_map,
    K_salient_tokens=50,
    assistant_mask=assistant_mask
)

print("\n--- Top Salient Thoughts ---")
salient_thoughts = [(i, count) for i, count in enumerate(salient_histogram) if count > 0]
salient_thoughts.sort(key=lambda x: x[1], reverse=True)

for i, count in salient_thoughts[:5]:  # Show top 5
    thought_text = get_thought_text(tokens, thoughts_token_map[i])
    print(f"Thought {i}: {count} salient tokens - {thought_text}")

# Create visualizations
plt.figure(figsize=(12, 8))

# 1. Plot thought interactions as a heatmap
if assistant_thoughts:
    thought_interaction_matrix = np.zeros((len(assistant_thoughts), len(assistant_thoughts)))
    
    for i, current_thought in enumerate(assistant_thoughts):
        for j, prev_thought in enumerate(assistant_thoughts):
            if current_thought in all_interactions and prev_thought in all_interactions[current_thought]:
                thought_interaction_matrix[i, j] = all_interactions[current_thought][prev_thought]['mean_all_scores']
    
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    im = plt.imshow(thought_interaction_matrix, cmap='viridis')
    plt.colorbar(label='Mean Attention Score')
    plt.title('Thought Interactions (Assistant Only)')
    plt.xlabel('Previous Thought Index')
    plt.ylabel('Current Thought Index')
    
    # Add grid lines to separate thoughts
    for i in range(len(assistant_thoughts)-1):
        # Horizontal lines
        plt.axhline(y=i+0.5, color='white', linestyle='-', linewidth=0.5)
        # Vertical lines
        plt.axvline(x=i+0.5, color='white', linestyle='-', linewidth=0.5)
    
    # Add thought indices as ticks
    plt.xticks(range(len(assistant_thoughts)), assistant_thoughts)
    plt.yticks(range(len(assistant_thoughts)), assistant_thoughts)
    
    # Add thought index labels inside the cells for better readability
    for i in range(len(assistant_thoughts)):
        for j in range(len(assistant_thoughts)):
            if thought_interaction_matrix[i, j] > 0.001:  # Only add text for significant interactions
                text_color = 'white' if thought_interaction_matrix[i, j] > 0.01 else 'black'
                plt.text(j, i, f"{assistant_thoughts[i]},{assistant_thoughts[j]}", 
                         ha="center", va="center", color=text_color, fontsize=6)

# 2. Plot salient token distribution
plt.subplot(1, 2, 2)
plt.bar(range(len(salient_histogram)), salient_histogram)
plt.xlabel('Thought Index')
plt.ylabel('Salient Token Count')
plt.title('Distribution of Salient Tokens')

if assistant_thoughts:
    plt.axvline(x=assistant_thoughts[0], color='r', linestyle='--', label='Start of Assistant Response')
    plt.legend()

plt.tight_layout()
plt.savefig('qwen_attention_analysis.png')
print("\nVisualization saved as 'qwen_attention_analysis.png'") 