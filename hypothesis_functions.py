import numpy as np
import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import shutil
from dataclasses import dataclass
from utils.hypothesis_utils import (
    identify_assistant_tokens,
    identify_thoughts,
    identify_salient_thoughts,
    calculate_thought_interactions,
    print_interactions    
)

def init_output_dir(output_dir: str, overwrite_output_dir: bool) -> None:
    """
    Initializes the output directory. If overwrite_output_dir is True and
    the directory exists, it will be emptied.

    Args:
        output_dir (str): Path to the output directory.
        overwrite_output_dir (bool): Whether to clear the directory if it exists.
    """
    if os.path.exists(output_dir):
        if overwrite_output_dir:
            # Clear the directory
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        else:
            # Leave it as is
            pass
    else:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir)

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

@dataclass
class Result:
    """
    A class to hold the results of the hypothesis analysis.
        all_interactions: dict 
            A dictionary containing the interaction statistics for each thought as explained in calculate_thought_interaction
        dict_result_all_context_windows: dict
            A dictionary containing 4D numpy arrays for each context window, 
            with mean attention scores for each thought interaction across all heads and layers.
            (extracted from all_interactions mean_all_scores statistics)
        salient_thoughts_all_heads_all_layers_array: np.ndarray
            A 3D numpy array containing the salient thoughts for each head and layer.
            (extracted from the )
    """
    all_interactions: dict
    dict_result_all_context_windows: dict
    salient_thoughts_all_heads_all_layers_array: np.ndarray
    

def hypothesis_run(model: AutoModelForCausalLM, 
                   tokenizer: AutoTokenizer,  
                   sequence_path: str, 
                   log_filename_thought_interactions: str, 
                   log_filename_salient_thoughts: str,
                   context_windows = [-1, 5, 10, 15]
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
        salient_thoughts_all_heads_all_layers_array = np.zeros((num_i, num_j, N))

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
                salient_thoughts_all_heads_all_layers_array[i//4, head] = salient_thoughts
                print(f"\thead {head}: {salient_thoughts}")

    finally:
        # Restore original stdout and close log file
        restore_logging(original_stdout, log_file)
        print(f"Logs written to {log_filename_salient_thoughts}")

    original_stdout, log_file = setup_logging(log_filename_thought_interactions)
    
    thought_interaction_matrix_mean_attn_scores = {i: {j: np.zeros((len(thoughts_token_map), len(thoughts_token_map))) for j in range(forward_output.attentions[0].shape[1])} \
                                                    for i in range(3, len(forward_output.attentions), 4)}
    thought_interaction_matrix_mean_topk_attn_scores = {i: {j: np.zeros((len(thoughts_token_map), len(thoughts_token_map))) for j in range(forward_output.attentions[0].shape[1])} \
                                                    for i in range(3, len(forward_output.attentions), 4)}
    thought_interaction_matrix_mean_attn_scores_array = np.zeros((num_i, num_j, N, N))
    thought_interaction_matrix_mean_topk_attn_scores_array = np.zeros((num_i, num_j, N, N))
    
    dict_result_all_context_windows = {}
    try:        
        # Initialize the dictionary for the current context window
        all_interactions = {context_window: {layer: {head: {} for head in range(forward_output.attentions[layer].shape[1])} for layer in range(3, len(forward_output.attentions), 4)} for context_window in context_windows}  # Initialize the dictionary for all interactions
        # draw stats for each context window
        for context_window in context_windows: 
            for layer in range(3, len(forward_output.attentions), 4):
                for head in range(forward_output.attentions[layer].shape[1]):
                    # Extract the attention matrix for the layer and head
                    curr_layer_curr_head_attention = forward_output.attentions[layer][0, head]
                    # Thought interaction analysis for the thoughts
                    for thought_idx in range(1, len(thoughts)):
                        interactions = calculate_thought_interactions(
                            curr_layer_curr_head_attention,
                            thoughts_token_map,
                            thought_idx,
                            K=15,
                            context_window=context_window,
                        )
                        all_interactions[context_window][layer][head][thought_idx] = interactions
                        # Print the interactions for the current thought
                        print_interactions(thought_idx, interactions, context_window)
                    
                    for i in range(len(thoughts_token_map)):
                        for j in range(len(thoughts_token_map)):
                            if i > j:
                                thought_interaction_matrix_mean_attn_scores[layer][head][i, j] = all_interactions[context_window][layer][head][i][j]['mean_all_scores_mean']
                                thought_interaction_matrix_mean_topk_attn_scores[layer][head][i, j] = all_interactions[context_window][layer][head][i][j]['mean_top_k_scores_mean']
                    thought_interaction_matrix_mean_attn_scores_array[layer//4, head] = thought_interaction_matrix_mean_attn_scores[layer][head]
                    thought_interaction_matrix_mean_topk_attn_scores_array[layer//4, head] = thought_interaction_matrix_mean_topk_attn_scores[layer][head]
            
            dict_result_all_context_windows[context_window] = {'mean_all_scores': thought_interaction_matrix_mean_attn_scores_array, \
                                                            'mean_top_k_scores': thought_interaction_matrix_mean_topk_attn_scores_array}
        
    finally: 
        # Restore original stdout and close log file
        restore_logging(original_stdout, log_file)
        print(f"Logs written to {log_filename_thought_interactions}")

    return Result(
            all_interactions=all_interactions,
            dict_result_all_context_windows=dict_result_all_context_windows,
            salient_thoughts_all_heads_all_layers_array=salient_thoughts_all_heads_all_layers_array
        )
            

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
    
