import numpy as np
import matplotlib.pyplot as plt
from analp_functions_for_analysis import (
    identify_assistant_tokens,
    calculate_thought_interaction,
    identify_salient_thoughts
)

def simple_tokenize(text):
    """Very basic tokenization by splitting on whitespace and punctuation."""
    # Replace some punctuation with spaces to ensure they're separate tokens
    for char in ".,()[]{}=+-*/^":
        text = text.replace(char, f" {char} ")
    # Split on whitespace
    tokens = text.split()
    return tokens

def identify_thoughts(tokens, markers=None):
    """
    Group tokens into thought chunks based on markers or sentence boundaries.
    For this example, we'll use periods as thought boundaries.
    """
    if markers is None:
        # Default: split at periods, question marks, exclamation points
        markers = ['.', '?', '!']
    
    thoughts = []
    current_thought = []
    
    for i, token in enumerate(tokens):
        current_thought.append(i)  # Add token index to current thought
        
        # If token ends with a marker or is a marker, end the current thought
        if any(token.endswith(m) for m in markers) or token in markers:
            if current_thought:  # Only add non-empty thoughts
                thoughts.append(current_thought)
                current_thought = []
    
    # Add the last thought if not empty
    if current_thought:
        thoughts.append(current_thought)
    
    return thoughts

def generate_mock_attention_matrix(num_tokens):
    """
    Generate a mock triangular attention matrix based on the number of tokens.
    Each token attends to itself and previous tokens only (causal attention).
    Each row is normalized to sum to 1.
    """
    # Initialize with zeros
    attention_matrix = np.zeros((num_tokens, num_tokens))
    
    # Fill in the lower triangle with decreasing attention based on distance
    for i in range(num_tokens):
        # For each token position i, generate attention to positions 0 through i
        # Closer tokens get higher attention values
        raw_attention = np.zeros(i + 1)
        for j in range(i + 1):
            # Attention decays with distance (i - j), but self-attention (i = j) is high
            if i == j:
                raw_attention[j] = 0.5  # High self-attention
            else:
                # Attention decays with distance, but recent tokens get more attention
                distance = i - j
                raw_attention[j] = 1.0 / (1.0 + distance)
        
        # Normalize to sum to 1
        if raw_attention.sum() > 0:
            raw_attention = raw_attention / raw_attention.sum()
        
        # Assign to the matrix
        attention_matrix[i, :i+1] = raw_attention
    
    return attention_matrix

def get_thought_text(tokens, thought_indices, max_tokens=20):
    """Get a readable representation of a thought, with truncation if needed."""
    thought_tokens = [tokens[idx] for idx in thought_indices]
    text = ' '.join(thought_tokens[:max_tokens])
    if len(thought_tokens) > max_tokens:
        text += '...'
    return text

# Read the Qwen2.5 file
with open('Qwen2.5-Math_math-algebra_2_2shot_full_sequence.txt', 'r') as f:
    content = f.read()

# Basic tokenization
tokens = simple_tokenize(content)
print(f"Total tokens: {len(tokens)}")

# Identify which tokens are from the assistant response
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

# Generate a mock attention matrix
attention_matrix = generate_mock_attention_matrix(len(tokens))
print(f"Generated attention matrix of shape: {attention_matrix.shape}")

# Calculate token salience scores (using row sums of attention matrix)
token_salience_scores = attention_matrix.sum(axis=0)

# Print examples of the most important thoughts (first few and some assistant thoughts)
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
# Collect data for all assistant thoughts
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
        
        # Find the most significant previous thought (highest mean_all_scores)
        most_significant = None
        highest_score = 0
        for prev_idx, scores in interactions.items():
            if scores['mean_all_scores'] > highest_score:
                highest_score = scores['mean_all_scores']
                most_significant = prev_idx
        
        if most_significant is not None and highest_score > 0.001:
            print(f"\nThought {i} most attends to Thought {most_significant} (score: {highest_score:.4f}):")
            current_text = get_thought_text(tokens, thoughts_token_map[i])
            prev_text = get_thought_text(tokens, thoughts_token_map[most_significant])
            print(f"  Current: {current_text}")
            print(f"  Attends to: {prev_text}")

# Identify salient thoughts
salient_histogram = identify_salient_thoughts(
    token_salience_scores,
    thoughts_token_map,
    K_salient_tokens=50,
    assistant_mask=assistant_mask
)

print("\n--- Top Salient Thoughts ---")
# Only show thoughts with at least one salient token
salient_thoughts = [(i, count) for i, count in enumerate(salient_histogram) if count > 0]
# Sort by count (descending)
salient_thoughts.sort(key=lambda x: x[1], reverse=True)

for i, count in salient_thoughts[:5]:  # Show top 5
    thought_text = get_thought_text(tokens, thoughts_token_map[i])
    print(f"Thought {i}: {count} salient tokens - {thought_text}")

# Create a visualization of the attention pattern
plt.figure(figsize=(12, 8))

# 1. Plot thought interactions as a heatmap
if assistant_thoughts:
    # Matrix for interaction between assistant thoughts
    thought_interaction_matrix = np.zeros((len(assistant_thoughts), len(assistant_thoughts)))
    
    for i, current_thought in enumerate(assistant_thoughts):
        for j, prev_thought in enumerate(assistant_thoughts):
            if current_thought in all_interactions and prev_thought in all_interactions[current_thought]:
                thought_interaction_matrix[i, j] = all_interactions[current_thought][prev_thought]['mean_all_scores']
    
    plt.subplot(1, 2, 1)
    plt.imshow(thought_interaction_matrix, cmap='viridis')
    plt.colorbar(label='Mean Attention Score')
    plt.title('Thought Interactions (Assistant Only)')
    plt.xlabel('Previous Thought Index')
    plt.ylabel('Current Thought Index')
    
    # Use actual thought indices as ticks
    plt.xticks(range(len(assistant_thoughts)), assistant_thoughts)
    plt.yticks(range(len(assistant_thoughts)), assistant_thoughts)

# 2. Plot salient token distribution
plt.subplot(1, 2, 2)
plt.bar(range(len(salient_histogram)), salient_histogram)
plt.xlabel('Thought Index')
plt.ylabel('Salient Token Count')
plt.title('Distribution of Salient Tokens')

# Add a vertical line at the first assistant thought
if assistant_thoughts:
    plt.axvline(x=assistant_thoughts[0], color='r', linestyle='--', label='Start of Assistant Response')
    plt.legend()

plt.tight_layout()
plt.savefig('thought_analysis.png')
print("\nVisualization saved as 'thought_analysis.png'") 