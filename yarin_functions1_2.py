import numpy as np

def identify_assistant_tokens(tokens):
    """
    Identifies which tokens appear after the word 'assistant' in a sequence.
    
    Args:
        tokens (list): A list of token strings.
        
    Returns:
        list: A boolean mask where True indicates tokens after 'assistant', 
              and False indicates tokens before or including 'assistant'.
    """
    assistant_mask = [False] * len(tokens)
    
    # Find the index of 'assistant' token
    assistant_idx = -1
    for i, token in enumerate(tokens):
        if token.lower() == 'assistant':
            assistant_idx = i
            break
    
    # If 'assistant' is found, mark all subsequent tokens as True
    if assistant_idx != -1:
        for i in range(assistant_idx + 1, len(tokens)):
            assistant_mask[i] = True
            
    return assistant_mask

def calculate_thought_interaction(attention_matrix, thoughts_token_map, current_thought_idx, K, 
                                 assistant_mask=None):
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
        assistant_mask (list of bool, optional): A boolean mask where True indicates tokens after 
                                                'assistant', and False indicates tokens before or 
                                                including 'assistant'. If provided, attention will 
                                                only be calculated for tokens marked as True.

    Returns:
        dict: A dictionary where keys are indices of previous thoughts (int) and values are
              dictionaries with two keys:
              - 'mean_top_k_scores' (list of float): A list of K mean scores.
                                                     Empty if K=0 or if issues arise.
              - 'mean_all_scores' (float): The overall mean of mean attention scores.
                                           Defaults to 0.0 if issues arise.
              Returns an empty dict if current_thought_idx is invalid or the current thought has no tokens.
    """
    if not (0 <= current_thought_idx < len(thoughts_token_map)):
        print(f"Error: current_thought_idx {current_thought_idx} is out of bounds for thoughts_token_map.")
        return {}

    current_thought_token_indices = thoughts_token_map[current_thought_idx]
    if not current_thought_token_indices:
        # print(f"Warning: Current thought {current_thought_idx} has no tokens.")
        return {}

    # If assistant_mask is provided, apply it to attention_matrix
    if assistant_mask is not None:
        # Create a copy of the attention matrix to avoid modifying the original
        modified_attention_matrix = attention_matrix.copy()
        
        # For each token, if it's before 'assistant', zero out its attention
        for i in range(len(assistant_mask)):
            if not assistant_mask[i]:
                modified_attention_matrix[:, i] = 0.0
                
        # Use the modified attention matrix for calculations
        attention_matrix = modified_attention_matrix

    interaction_results = {}

    for prev_thought_idx in range(current_thought_idx):
        if not (0 <= prev_thought_idx < len(thoughts_token_map)):
            print(f"Warning: prev_thought_idx {prev_thought_idx} seems invalid. Skipping.")
            continue
            
        prev_thought_token_indices = thoughts_token_map[prev_thought_idx]
        
        mean_top_k_for_prev_thought = [0.0] * K if K > 0 else []
        mean_all_for_prev_thought = 0.0

        if not prev_thought_token_indices:
            interaction_results[prev_thought_idx] = {
                'mean_top_k_scores': mean_top_k_for_prev_thought,
                'mean_all_scores': mean_all_for_prev_thought
            }
            continue

        list_of_top_k_vectors_for_current_thought = []
        list_of_mean_scores_for_current_thought = []

        for token_idx_in_current_thought in current_thought_token_indices:
            if not (0 <= token_idx_in_current_thought < attention_matrix.shape[0]):
                print(f"Warning: Token index {token_idx_in_current_thought} from current thought is out of bounds for attention matrix row. Skipping token.")
                if K > 0: # Still add default for averaging later
                    list_of_top_k_vectors_for_current_thought.append([0.0] * K)
                list_of_mean_scores_for_current_thought.append(0.0)
                continue

            valid_prev_thought_indices = [idx for idx in prev_thought_token_indices if 0 <= idx < attention_matrix.shape[1]]
            if not valid_prev_thought_indices:
                if K > 0:
                    list_of_top_k_vectors_for_current_thought.append([0.0] * K)
                list_of_mean_scores_for_current_thought.append(0.0)
                continue

            attn_scores_from_current_token_to_prev_thought = attention_matrix[token_idx_in_current_thought, valid_prev_thought_indices]
            
            current_token_top_k_list = [0.0] * K if K > 0 else []
            current_token_mean_all = 0.0

            if attn_scores_from_current_token_to_prev_thought.size > 0:
                if K > 0:
                    actual_k_for_token = min(K, len(attn_scores_from_current_token_to_prev_thought))
                    top_k_values_for_token = np.sort(attn_scores_from_current_token_to_prev_thought)[::-1][:actual_k_for_token]
                    current_token_top_k_list[:actual_k_for_token] = top_k_values_for_token
                current_token_mean_all = np.mean(attn_scores_from_current_token_to_prev_thought)
            
            if K > 0:
                list_of_top_k_vectors_for_current_thought.append(current_token_top_k_list)
            list_of_mean_scores_for_current_thought.append(current_token_mean_all)

        if K > 0:
            if list_of_top_k_vectors_for_current_thought:
                mean_top_k_for_prev_thought = np.mean(np.array(list_of_top_k_vectors_for_current_thought), axis=0).tolist()

        if list_of_mean_scores_for_current_thought:
            mean_all_for_prev_thought = np.mean(list_of_mean_scores_for_current_thought)

        interaction_results[prev_thought_idx] = {
            'mean_top_k_scores': mean_top_k_for_prev_thought,
            'mean_all_scores': mean_all_for_prev_thought
        }
    return interaction_results


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
    if not isinstance(token_salience_scores, (np.ndarray, list)):
        print("Error: token_salience_scores must be a non-empty NumPy array or list.")
        return []
    if not thoughts_token_map:
        print("Error: thoughts_token_map cannot be empty.")
        return []
    
    salience_scores_np = np.array(token_salience_scores)
    total_tokens = len(salience_scores_np)

    if total_tokens == 0: # Should be caught by the first check if list, but good for ndarray
        print("Error: token_salience_scores is empty.")
        return [0] * len(thoughts_token_map) # Consistent return type

    # If assistant_mask is provided, zero out the salience scores for tokens before 'assistant'
    if assistant_mask is not None:
        modified_salience_scores = salience_scores_np.copy()
        for i in range(len(assistant_mask)):
            if not assistant_mask[i]:
                modified_salience_scores[i] = 0.0
        salience_scores_np = modified_salience_scores

    if K_salient_tokens <= 0:
        # print("Warning: K_salient_tokens is not positive. Returning zero counts per thought.")
        return [0] * len(thoughts_token_map)

    max_token_idx_in_map = -1
    if thoughts_token_map: # Check if not empty before iterating
        for thought_tokens in thoughts_token_map:
            if thought_tokens: # Check if thought itself is not empty
                current_max = max(thought_tokens)
                if current_max > max_token_idx_in_map:
                    max_token_idx_in_map = current_max
    
    if max_token_idx_in_map >= total_tokens:
        print(f"Error: Max token index in thoughts_token_map ({max_token_idx_in_map}) exceeds "
              f"length of token_salience_scores ({total_tokens}).")
        return []

    actual_K_salient = min(K_salient_tokens, total_tokens)
    
    if actual_K_salient > 0:
        indices_of_top_k_salient_tokens = np.argsort(salience_scores_np)[-actual_K_salient:][::-1]
    else:
        indices_of_top_k_salient_tokens = np.array([], dtype=int)

    salient_tokens_per_thought_histogram = [0] * len(thoughts_token_map)
    set_of_salient_token_indices = set(indices_of_top_k_salient_tokens)

    for i, thought_tokens_indices in enumerate(thoughts_token_map):
        count = 0
        for token_idx in thought_tokens_indices:
            if token_idx in set_of_salient_token_indices:
                count += 1
        salient_tokens_per_thought_histogram[i] = count

    return salient_tokens_per_thought_histogram
