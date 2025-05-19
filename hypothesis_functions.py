import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

def setup_logging(log_filename):
    """
    Sets up logging to redirect all prints to a log file.
    
    Args:
        log_filename (str): Name of the log file to write to
    
    Returns:
        The original stdout for restoration if needed
    """
    # Ensure logs directory exists
    log_dir = os.path.dirname(log_filename)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Open log file and redirect stdout
    log_file = open(log_filename, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = log_file
    
    return original_stdout, log_file

def restore_logging(original_stdout, log_file):
    """
    Restores original stdout and closes the log file.
    """
    sys.stdout = original_stdout
    log_file.close()

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

def calculate_thought_interaction(attention_matrix, thoughts_token_map, current_thought_idx, K):
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

    Returns:
        dict: A dictionary where keys are indices of previous thoughts (int) and values are
              dictionaries with two keys:
              - 'mean_top_k_scores' (list of float): A list of K mean scores.
                                                     Empty if K=0 or if issues arise.
              - 'mean_all_scores' (float): The overall mean of mean attention scores.
                                           Defaults to 0.0 if issues arise.
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

        for token_idx_in_current_thought in current_thought_token_indices:
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
        mean_all_for_prev_thought = torch.mean(torch.tensor(list_of_mean_scores_for_current_thought)).item()
        flat_list_of_indices_top_k_tokens_for_current_thought = [idx for sublist in list_of_indices_top_k_tokens_for_current_thought for idx in sublist]
        counts = Counter(flat_list_of_indices_top_k_tokens_for_current_thought)
        most_common_token_indices_for_prev_thought = counts.most_common(K)

        interaction_results[prev_thought_idx] = {
            'mean_top_k_scores': mean_top_k_for_prev_thought,
            'mean_all_scores': mean_all_for_prev_thought,
            'most_common_token_indices': most_common_token_indices_for_prev_thought
        }
    return interaction_results

# Function to visually print the interactions comfortably
def print_interactions(thought_idx, interactions):
    print(f"Thought interactions for thought {thought_idx}:")
    for i, interaction in interactions.items():
        print(f"Thought {i} with {thought_idx} -")
        print(f"mean_top_k_scores: {interaction['mean_top_k_scores']}")
        print(f"mean_all_scores: {interaction['mean_all_scores']}")
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


def hypothesis_run(model: AutoModelForCausalLM, 
                   tokenizer: AutoTokenizer,  
                   sequence_path: str, 
                   log_filename_thought_interactions: str, 
                   log_filename_salient_thoughts: str,
                ):
    """
        Main function to run the hypothesis analysis.
    """
    # get the folder path of the log file
    folder_path = os.path.dirname(log_filename_salient_thoughts)
    # Setup logging to redirect prints to file
    original_stdout, log_file = setup_logging(log_filename_salient_thoughts)
    
    try:        
        # Load the full sequence
        if os.path.exists(sequence_path):
            with open(sequence_path, 'r', encoding='utf-8') as f:
                full_sequence = f.read()
            print(f"Loaded sequence with length: {len(full_sequence)}")
        
        # Tokenize the sequence into words (simple tokenization for example)
        tokens = tokenizer.encode(full_sequence, add_special_tokens=True, return_tensors='pt').to('cuda:0')
        text_tokens = tokenizer.tokenize(full_sequence, add_special_tokens=True)
        print(f"Number of tokens: {len(tokens[0])}")
        
        # run forward pass to get the attention matrices
        with torch.no_grad():
            forward_output = model(tokens, output_attentions=True)

        # Identify which tokens are from the assistant
        assistant_token_id = tokenizer.convert_tokens_to_ids("assistant")
        assistant_mask = identify_assistant_tokens(tokens, assistant_token_id)
        print(f"Found {torch.sum(assistant_mask)} tokens after 'assistant'")

        text_tokens = text_tokens[-torch.sum(assistant_mask):]
        thoughts, thoughts_token_map = identify_thoughts(text_tokens, markers=tokenizer.tokenize("\n\n"))
        
        # Compute sizes
        num_i = len(range(3, len(forward_output.attentions), 4))
        num_j = forward_output.attentions[0].shape[1]
        N = len(thoughts_token_map)
        salient_thoughts_all_heads_all_layers = np.zeros((num_i, num_j, N))

        for i in range(3, len(forward_output.attentions), 4):
            print(f"Salient tokens distribution layer {i}")
            for head in range(forward_output.attentions[i].shape[1]):
                # Extract the attention matrix for the layer and head
                curr_layer_curr_head_attention = forward_output.attentions[i][0, head]  # Assuming the last layer and head are of interest
                
                # Example usage of functions
                # mask out the user's prompt attention and tokens
                curr_layer_curr_head_attention = curr_layer_curr_head_attention[-torch.sum(assistant_mask):, -torch.sum(assistant_mask):]
            
                # Calculate token salience (simple example)
                token_salience = torch.sum(curr_layer_curr_head_attention, axis=0)  # Sum of attention received by each token
                # normalize the salience scores
                non_zero_count = (token_salience != 0).sum(dim=0)
                normalized_token_salience = torch.where(non_zero_count > 0, token_salience / non_zero_count, torch.zeros_like(token_salience))

                max_length_thought = min(max([len(thought) for thought in thoughts]), 100)
                salient_thoughts = identify_salient_thoughts(normalized_token_salience, thoughts_token_map, max_length_thought, assistant_mask)
                salient_thoughts_all_heads_all_layers[i//4, head] = salient_thoughts
                print(f"\thead {head}: {salient_thoughts}")
                
        # save the salient thoughts
        np.save(os.path.join(folder_path, f"salient_thoughts.npy"), salient_thoughts_all_heads_all_layers)

    finally:
        # Restore original stdout and close log file
        restore_logging(original_stdout, log_file)
        print(f"Logs written to {log_filename_salient_thoughts}")

    original_stdout, log_file = setup_logging(log_filename_thought_interactions)
    thought_interaction_matrix_mean_attn_scores = {i: {j: np.zeros((len(thoughts_token_map), len(thoughts_token_map))) for j in range(forward_output.attentions[0].shape[1])} \
                                                    for i in range(3, len(forward_output.attentions), 4)}
    thought_interaction_matrix_mean_attn_scores_array = np.zeros((num_i, num_j, N, N))

    try:
        for layer in range(3, len(forward_output.attentions), 4):
            for head in range(forward_output.attentions[layer].shape[1]):
                # Extract the attention matrix for the layer and head
                curr_layer_curr_head_attention = forward_output.attentions[layer][0, head]
                all_interactions = {}
                # Thought interaction analysis for the thoughts
                for thought_idx in range(1, len(thoughts)):
                    interactions = calculate_thought_interaction(
                        curr_layer_curr_head_attention,
                        thoughts_token_map,
                        thought_idx,
                        K=15
                    )
                    all_interactions[thought_idx] = interactions
                    # Print the interactions for the current thought
                    print_interactions(thought_idx, interactions)
                
                for i in range(len(thoughts_token_map)):
                    for j in range(len(thoughts_token_map)):
                        if i > j:
                            thought_interaction_matrix_mean_attn_scores[layer][head][i, j] = all_interactions[i][j]['mean_all_scores']
                thought_interaction_matrix_mean_attn_scores_array[layer//4, head] = thought_interaction_matrix_mean_attn_scores[layer][head]
        
        # Save the thought interaction matrix
        np.save(os.path.join(folder_path, f"thought_interaction_matrix_mean_attn_scores.npy"), thought_interaction_matrix_mean_attn_scores_array)
        
    finally:
        # Restore original stdout and close log file
        restore_logging(original_stdout, log_file)
        print(f"Logs written to {log_filename_thought_interactions}")
        
    return all_interactions, thought_interaction_matrix_mean_attn_scores
            

# if __name__ == "__main__":
#     # Paths to the files
#     log_filename_thought_interactions = 'logs/Qwen2.5-Math_math-algebra_2_2shot/thought_interactions.txt'
#     log_filename_salient_thoughts = 'logs/Qwen2.5-Math_math-algebra_2_2shot/salient_thoughts.txt'
#     sequence_path = 'full_sequences/Qwen2.5-Math_math-algebra_2_2shot.txt'
    
#     # load tokenizer of Qwen/Qwen2.5-Math-7B-Instruct
#     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-7B-Instruct")
#     # load model
#     model = AutoModelForCausalLM.from_pretrained(
#                                             "Qwen/Qwen2.5-Math-7B-Instruct", 
#                                             # output_attentions=True, 
#                                             # return_dict_in_generate=True,
#                                             # trust_remote_code=True,
#                                             torch_dtype="auto",
#                                             # device_map="auto",
#                                             # low_cpu_mem_usage=True,
#                                             # attn_implementation="flash_attention_2" 
#                                         ).to('cuda:0')
#     all_interactions, thought_interaction_matrix_mean_attn_scores = hypothesis_run(
#                 model,
#                 tokenizer,
#                 sequence_path=sequence_path,
#                 log_filename_thought_interactions=log_filename_thought_interactions,
#                 log_filename_salient_thoughts=log_filename_salient_thoughts
#             )
    
