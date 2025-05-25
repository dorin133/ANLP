import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

def extract_assistant_thoughts_with_token_indices(file_path, model_name="Qwen/Qwen2.5-Math-7B-Instruct"):
    """
    Extracts assistant's thoughts and their token index ranges using a tokenizer.

    Args:
        file_path (str): Path to the input .txt file.
        model_name (str): Pretrained tokenizer to use from Hugging Face.

    Returns:
        List[dict]: A list of dicts, each with 'text', 'start_token_idx', and 'end_token_idx'.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the starting point of assistant text
    try:
        assistant_start = content.index("assistant")
    except ValueError:
        raise ValueError("'assistant' keyword not found in the file.")

    full_tokens = tokenizer.encode(content, add_special_tokens=False)
    full_encoded = tokenizer(content, return_offsets_mapping=True, add_special_tokens=False)
    offsets = full_encoded['offset_mapping']

    # Slice content starting from 'assistant' and split by double newlines
    assistant_text = content[assistant_start:]
    local_blocks = [block.strip() for block in assistant_text[len("assistant"):].strip().split("\n\n") if block.strip()]

    # Create thought metadata with token index ranges
    thoughts_with_indices = []
    for block in local_blocks:
        # Find where this block appears in the original content
        block_start_char = content.index(block)
        block_end_char = block_start_char + len(block)

        # Find the corresponding token indices
        start_token_idx = None
        end_token_idx = None
        for i, (start, end) in enumerate(offsets):
            if start_token_idx is None and start >= block_start_char:
                start_token_idx = i
            if end_token_idx is None and end > block_end_char:
                end_token_idx = i
                break
        # Handle case where the block is at the very end
        if end_token_idx is None:
            end_token_idx = len(offsets)

        thoughts_with_indices.append({
            "text": block,
            "start_token_idx": start_token_idx,
            "end_token_idx": end_token_idx
        })

    return thoughts_with_indices


def identify_thoughts(tokens, markers=None):
    """
    Group tokens into thought chunks based on markers or sentence boundaries.
    For this example, we'll use periods as thought boundaries.
    """
    if markers is None:
        # return error
        return "Error: No markers provided for thought segmentation."

    thoughts = []
    current_thought = []
    thoughts_indices_map = []  # To keep track of the indices of tokens in each thought
    current_thought_indices = []
    
    for i, token in enumerate(tokens):
        current_thought.append(token)  # Add token to current thought
        current_thought_indices.append(i)

        # If token ends with a marker or is a marker, end the current thought
        if any(token.endswith(m) for m in markers) or token in markers:
            if current_thought:  # Only add non-empty thoughts
                thoughts.append(current_thought)
                thoughts_indices_map.append(current_thought_indices)
                current_thought = []
                current_thought_indices = []
    
    # Add the last thought if not empty
    if current_thought:
        thoughts.append(current_thought)
        thoughts_indices_map.append(current_thought_indices)
    
    return thoughts, thoughts_indices_map

def identify_assistant_tokens(tokens, assistant_token_id):
    """
    Identifies which tokens appear after the word 'assistant' in a sequence.
    
    Args:
        tokens (list): A list of token strings.
        assistant_token_id (int): The token ID to identify the boundary. Default is 'assistant'.
    Returns:
        list: A boolean mask where True indicates tokens after 'assistant',
              and False indicates tokens before or including 'assistant'.
    """
    assistant_mask = torch.ones_like(tokens).to('cuda:0')

    # Find the index of 'assistant' token
    assistant_idx = -1
    assistant_tokens = (tokens == assistant_token_id).nonzero()
    
    # If 'assistant' is found, mark all subsequent tokens as True
    if assistant_tokens.numel() > 0:
        assistant_idx = assistant_tokens[0, 1].item()
        # Mark all tokens before assistant_idx as masked (FALSE==0)
        assistant_mask[:, :assistant_idx+1] = 0

    return assistant_mask

def calculate_thought_interactions(attention_matrix, thoughts_token_map, current_thought_idx, K, context_window=-1):
    """
    Calculates interaction scores between a current thought and all its preceding thoughts.

    The interaction is measured by:
    1. Mean of Top-K attention scores: For each token in the current thought, its Top-K attention
       scores (from its row in the attention matrix) towards tokens of a specific previous
       thought are found. These K-dimensional vectors of scores are then averaged (element-wise)
       across all tokens in the current thought.
    2. Mean of all attention scores: For each token in the current thought, its mean attention
       towards all tokens in a specific previous thought is calculated. These mean scores are
       then averaged across all tokens in the current thought.

    Args:
        attention_matrix (np.ndarray): A 2D NumPy array where attention_matrix[query_idx, key_idx]
                                       is the attention score from the query_token (at query_idx)
                                       to the key_token (at key_idx).
                                       The shape is (total_num_tokens, total_num_tokens).
        thoughts_token_map (list of list of int): A list where each inner list contains the
                                                 global token indices for a specific thought.
                                                 e.g., thoughts_token_map[0] = [0, 1, 2] (tokens for thought 0).
                                                 Indices must be valid for the attention_matrix.
        current_thought_idx (int): The index of the current thought (segment) to analyze.
                                   This corresponds to an index in `thoughts_token_map`.
        K (int): The number of top attention scores to consider for the Top-K metric.
                 If K=0, Top-K scores will be an empty list.
        context_window (int, -1 by default): The number of tokens to consider in the context window of the current thought 
                                        to iterate over for statistics of the previous thoughts.
                                        If -1, all the thought tokens are considered.
                                        

    Returns:
        dict: A dictionary where keys are indices of previous thoughts (int) and values are
              dictionaries with two keys:
              - 'mean_top_k_scores' (list of float): A list of K mean scores.
                                            Each element corresponds to the mean of Top-K attention
                                            scores for a specific token in the current thought.
                                            If K=0, this will be an empty list.
              - 'mean_all_scores' (float): The overall mean of mean attention scores.
                                            This is the mean of all attention scores from a  given
                                            current thought to each previous thought.
              - 'most_common_token_indices' (list of tuple): A list of tuples where each tuple
                                            contains a token index and its count in the Top-K scores.
                                            This is useful for identifying which tokens are most frequently attended to.
              Returns an empty dict if current_thought_idx is invalid or the current thought has no tokens.
    """
    if K <= 0:
        print("Error: K must be greater than 0 for Top-K scores.")
        return {}
    
    current_thought_token_indices = thoughts_token_map[current_thought_idx]
    interaction_results = {}

    for prev_thought_idx in range(current_thought_idx):
        prev_thought_token_indices = thoughts_token_map[prev_thought_idx]
        
        mean_top_k_for_prev_thought = [0.0] * K
        mean_all_for_prev_thought = 0.0

        list_of_top_k_vectors_for_current_thought = []
        list_of_indices_top_k_tokens_for_current_thought = []
        list_of_mean_scores_for_current_thought = []
        
        context_window_range = min(context_window, len(current_thought_token_indices))
        current_thought_context_window_indices = current_thought_token_indices if context_window == -1 else \
                        current_thought_token_indices[:context_window_range]

        for token_idx_in_current_thought in current_thought_context_window_indices:
            attn_scores_from_current_token_to_prev_thought = attention_matrix[token_idx_in_current_thought, prev_thought_token_indices]
            
            current_token_top_k_list = [0.0] * K
            current_token_mean_all = 0.0
            current_token_indices_top_k_list = [0] * K

            actual_k_for_token = min(K, len(attn_scores_from_current_token_to_prev_thought))
            top_k_values_for_token = torch.sort(attn_scores_from_current_token_to_prev_thought, descending=True)[0][:actual_k_for_token]
            current_token_indices_top_k_list[:actual_k_for_token] = torch.sort(attn_scores_from_current_token_to_prev_thought, descending=True)[1][:actual_k_for_token].tolist()
            current_token_top_k_list[:actual_k_for_token] = top_k_values_for_token
            current_token_mean_all = torch.mean(attn_scores_from_current_token_to_prev_thought)
            
            list_of_top_k_vectors_for_current_thought.append(current_token_top_k_list)
            list_of_mean_scores_for_current_thought.append(current_token_mean_all)
            list_of_indices_top_k_tokens_for_current_thought.append(current_token_indices_top_k_list)

        mean_top_k_for_prev_thought = torch.mean(torch.tensor(list_of_top_k_vectors_for_current_thought), axis=0).tolist()
        top_k_mean_val_for_prev_thought = torch.mean(torch.tensor(list_of_top_k_vectors_for_current_thought)).item()
        mean_all_for_prev_thought = torch.mean(torch.tensor(list_of_mean_scores_for_current_thought)).item()
        flat_list_of_indices_top_k_tokens_for_current_thought = [idx for sublist in list_of_indices_top_k_tokens_for_current_thought for idx in sublist]
        counts = Counter(flat_list_of_indices_top_k_tokens_for_current_thought)
        most_common_token_indices_for_prev_thought = counts.most_common(K)

        interaction_results[prev_thought_idx] = {
            'mean_top_k_scores': mean_top_k_for_prev_thought,
            'mean_top_k_scores_mean': top_k_mean_val_for_prev_thought,
            'mean_all_scores_mean': mean_all_for_prev_thought,
            'most_common_token_indices': most_common_token_indices_for_prev_thought
        }
    return interaction_results

def compute_mean_std(arr: np.ndarray):
    """
    Compute per-group mean and standard deviation over axis 1 of a (n, m, x, y) array.

    Parameters:
        arr (np.ndarray): Input array of shape (n, m, x, y)

    Returns:
        means (np.ndarray): Mean values of shape (n, x, y)
        stds  (np.ndarray): Standard deviations of shape (n, x, y)
    """
    means = np.mean(arr, axis=1)  # Shape: (n, x, y)
    stds = np.std(arr, axis=1)    # Shape: (n, x, y)
    return means, stds


def process_and_save_mean_std(path_to_npy: str):
    """
    Load a .npy file of shape (n, m, x, y), compute mean and std over axis=1,
    and save the results as separate .npy files.

    Parameters:
        path_to_npy (str): Path to the input .npy file
    """
    # Load array
    arr = np.load(path_to_npy)  # shape assumed to be (n, m, x, y)

    # Validate shape
    if arr.ndim != 4:
        raise ValueError(f"Expected array of shape (n, m, x, y), got shape {arr.shape}")

    # Compute mean and std along axis=1
    means = np.mean(arr, axis=1)
    stds = np.std(arr, axis=1)

    # Build output file paths
    base, _ = os.path.splitext(path_to_npy)
    mean_path = f"{base}_mean_along_heads.npy"
    std_path = f"{base}_std_along_heads.npy"

    # Save results
    np.save(mean_path, means)
    np.save(std_path, stds)

    print(f"✅ Saved mean to {mean_path}")
    print(f"✅ Saved std to {std_path}")


# Function to visually print the interactions comfortably
def print_interactions(thought_idx, interactions, context_window):
    print(f"Thought interactions for thought {thought_idx}, and context window {context_window}:")
    for i, interaction in interactions.items():
        print(f"Thought {i} with {thought_idx} -")
        print(f"mean_top_k_scores: {interaction['mean_top_k_scores']}")
        print(f"mean_top_k_scores_mean: {interaction['mean_top_k_scores_mean']}")
        print(f"mean_all_scores_mean: {interaction['mean_all_scores_mean']}")
        print(f"most_common_token_indices: {interaction['most_common_token_indices']} \n")

def identify_salient_thoughts(token_salience_scores, thoughts_token_map, K_salient_tokens, 
                             assistant_mask=None):
    """
    Identifies thought-steps with the highest amount of salient tokens.

    Salient tokens are identified globally based on their pre-computed normalized attention scores.
    A histogram is built to count how many of these globally salient tokens fall into each thought.

    Args:
        token_salience_scores (np.ndarray or list): A 1D array or list of salience scores
                                                    for every token in the entire sequence.
        thoughts_token_map (list of list of int): A list where each inner list contains the
                                                 global token indices for a specific thought.
        K_salient_tokens (int): The number of globally top salient tokens to consider.
        assistant_mask (list of bool, optional): A boolean mask where True indicates tokens after 
                                                'assistant', and False indicates tokens before or 
                                                including 'assistant'. If provided, only tokens 
                                                marked as True will be considered for salience.

    Returns:
        list: A list representing the histogram. The value at index `i` is the count
              of salient tokens belonging to thought `i`.
              Returns an empty list if inputs are invalid.
    """


    # # If assistant_mask is provided, zero out the salience scores for tokens before 'assistant'
    # if assistant_mask is not None:
    #     modified_salience_scores = salience_scores_np.copy()
    #     for i in range(len(assistant_mask)):
    #         if not assistant_mask[i]:
    #             modified_salience_scores[i] = 0.0
    #     salience_scores_np = modified_salience_scores
    thoughts_combined_length = sum([len(thought) for thought in thoughts_token_map])
    if thoughts_combined_length == 0:
        print("Error: One or more thoughts have no tokens.")
        return []
    actual_K_salient = min(K_salient_tokens, thoughts_combined_length)
    
    indices_of_top_k_salient_tokens = torch.argsort(token_salience_scores, dim=-1, descending=False)[-actual_K_salient:].to('cpu').numpy()
    
    salient_tokens_per_thought_histogram = [0] * len(thoughts_token_map)

    for i, thought_tokens_indices in enumerate(thoughts_token_map):
        count = 0
        for token_idx in thought_tokens_indices:
            if token_idx in indices_of_top_k_salient_tokens:
                count += 1
        salient_tokens_per_thought_histogram[i] = count

    # divide by the length of the thought
    salient_tokens_per_thought_histogram = [count / len(thoughts_token_map[i]) for i, count in enumerate(salient_tokens_per_thought_histogram)]
    # normalize the histogram to sum to 1
    salient_tokens_per_thought_histogram = [count / sum(salient_tokens_per_thought_histogram) for count in salient_tokens_per_thought_histogram]
    
    return salient_tokens_per_thought_histogram
