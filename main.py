import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import seaborn as sns
import json
from datasets import concatenate_datasets
import numpy as np
from hypothesis_functions import hypothesis_run, init_output_dir
from utils.hypothesis_utils import process_mean_std
import matplotlib.pyplot as plt
from utils.visualize_salient_thoughts import visualize_thoughts_interactions, visualize_salient_thoughts
import copy
import sys

# parse arguments from the .sh script passed to main.py
import argparse
parser = argparse.ArgumentParser(description="Run the model with specified parameters.")
parser.add_argument(
    "--max_new_tokens", 
    type=int, 
    default=512, 
    help="Maximum number of new tokens to generate."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./output_fixed",
    help="Directory to save all outputs of the current run."
)
parser.add_argument(
    "--num_samples_per_task",
    type=int,
    default=2,
    help="Number of samples to generate per task."
)
parser.add_argument(
    "--sample_indices",
    type=str,
    default="[1, 2, 3]",
    help="Indices of the samples to process over the chosen dataset"
)
# parse argument of the context windows list
parser.add_argument(
    "--context_windows",
    type=str,
    default="[-1, 3, 6]",
    help="List of context windows to use. Meaning number of tokens to consider for the statistics. of the current thought."
)
parser.add_argument(
    "--dataset_name",
    type=str,
    default="math-algebra",
    help="Name of the dataset to use. Must be one of the available datasets."
)
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen3",
    help="Name of the model to use. Must be one of the available models."
)
args = parser.parse_args()
# Convert the context windows and sample indices strings to lists of integers
args.context_windows = eval(args.context_windows)
args.sample_indices = eval(args.sample_indices)

# Configuration
models = {
    "Qwen2.5-Math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Qwen3": "Qwen/Qwen3-8B"
}

# Check if the specified model exists
if args.model_name not in models:
    print(f"Error: Model '{args.model_name}' not found in available models.")
    print(f"Available models: {list(models.keys())}")
    sys.exit(1)

# Filter models to only include the specified model
selected_model = {args.model_name: models[args.model_name]}

hendrycks_math_names = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

datasets = {
    # "gsm8k": load_dataset("openai/gsm8k", "main", split="test"),
        # "math-algebra": {
        #     "dataset": load_dataset(
        #         "EleutherAI/hendrycks_math", 
        #         hendrycks_math_names[0], 
        #         split="test[:100]", 
        #         trust_remote_code=True
        #     ).shuffle(seed=42).select(args.sample_indices),  # Select specific sample by index
        #     "config": "0-shot"
        # },
        "aime2024": {
            "dataset": concatenate_datasets(
                [
                    load_dataset
                    (
                        "Maxwell-Jia/AIME_2024",
                        split="train",
                        trust_remote_code=True
                    )
                ]
            ).shuffle(seed=42).select(args.sample_indices),
            "config": "0-shot"
        },
}

# Check if the specified dataset exists
if args.dataset_name not in datasets:
    print(f"Error: Dataset '{args.dataset_name}' not found in available datasets.")
    print(f"Available datasets: {list(datasets.keys())}")
    sys.exit(1)

# Filter to use only the specified dataset
dataset = datasets[args.dataset_name]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attention hook function
attention_scores = {}

# Prepare Chain-of-Thought prompt
def prepare_prompt(example, dataset_name, shot_examples=None):
    if dataset_name == "gsm8k":
        return f"Q: {example['question']}\n"
    elif ("math" in dataset_name):
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['problem']}\nFull Solution: {ex['solution']}\n" 
                                    for ex in shot_examples]
                                )
            prompt = f"{incontext}\n" + f"Question: {example['problem']}\nFull Solution:"
        else:
            prompt = f"Question: {example['problem']}\nFull Solution:"
        return prompt
    elif "aime" in dataset_name:
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['Problem']}\nFull Solution: {ex['Solution']}\n" 
                                    for ex in shot_examples]
                                )
            prompt = f"{incontext}\n" + f"Question: {example['Problem']}\nFull Solution:"
        else:
            prompt = f"Question: {example['Problem']}\nFull Solution:"
        return prompt
    else:
        return example['text']

# === Main evaluation loop ===
print(f"Loading {args.model_name}...")
tokenizer = AutoTokenizer.from_pretrained(selected_model[args.model_name], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
                            selected_model[args.model_name], 
                            # output_attentions=True, 
                            # return_dict_in_generate=True,
                            trust_remote_code=True,
                            torch_dtype="auto",
                            device_map="auto",
                            # low_cpu_mem_usage=True,
                            attn_implementation="eager" 
                        )
model.eval()

print(f"Evaluating on {args.dataset_name}...")
config = dataset.get("config", None)

for i, example in zip(args.sample_indices, dataset['dataset']):
    
    folder_path = f"{args.output_dir}/{args.model_name}_{args.dataset_name}_{i}_{config}"
        
    os.makedirs(folder_path, exist_ok=True)
    print(f"Processing example {i} for {args.model_name} on {args.dataset_name}...")
    
    shot_examples = None
    if config == "2-shot":
        # randomly select for sample indices in the dataset
        candidates = np.delete(np.arange(len(dataset['dataset'])), i)
        shot_examples_indices = np.random.choice(
            candidates,
            size=2,
            replace=False
        )
        shot_examples = dataset['dataset'].select(shot_examples_indices)
    prompt = prepare_prompt(example, args.dataset_name, shot_examples)
    messages = [
                # {"role": "system", "content": "Please reason step by step. Each thought should be separated by exactly two newline characters r'\n\n'. Put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt}
            ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        if os.path.exists(os.path.join(folder_path, "full_sequence.txt")):
            with open(os.path.join(folder_path, "full_sequence.txt"), 'r', encoding="utf-8") as f:
                full_sequence = f.read()
                output = tokenizer([full_sequence])['input_ids']
        else:
            output = model.generate(**inputs, 
                        max_new_tokens=max(min(args.max_new_tokens, model.config.max_position_embeddings-inputs['input_ids'].shape[1]),0),
                    )

    if not os.path.exists(os.path.join(folder_path, "full_sequence.txt")):
        # Save the entire sequence:
        with open(os.path.join(folder_path, "full_sequence.txt"), 'w', encoding="utf-8") as f:
            full_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
            f.write(full_sequence)  

    if not os.path.exists(os.path.join(folder_path, "generated_answer.txt")):
        # Save the model's generated answer:
        with open(os.path.join(folder_path, "generated_answer.txt"), 'w', encoding="utf-8") as f:
            generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
                    ]
            answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            f.write(answer)

    # run the hypothesis function on the unseen context windows
    result = hypothesis_run(
        model,
        tokenizer,
        sequence_path=os.path.join(folder_path, "full_sequence.txt"),
        log_filename_thought_interactions=os.path.join(folder_path, "thought_interactions.txt"),
        log_filename_salient_thoughts=os.path.join(folder_path, "salient_thoughts.txt"),
        context_windows=args.context_windows,
        K=15,  # Top-K interactions to consider
    )

    # Visualize the salient thoughts for the current context window
    visualize_salient_thoughts(data=result.salient_thoughts_all_heads_all_layers_array, \
                                title=f"Salient Thoughts - {args.model_name} {args.dataset_name} {i}",
                                output_dir=folder_path)
        
    # Save the salient thoughts (this result is independent of the context windows)
    np.save(os.path.join(folder_path, f"salient_thoughts.npy"), result.salient_thoughts_all_heads_all_layers_array)
    # Save the dictionary of the salient tokens details in result.salient_tokens_all_heads_all_layers_dicts
    
    #convert result.salient_tokens_all_heads_all_layers_dicts to numpy and save it as numpy array
    salient_tokens_all_heads_all_layers_dicts_array = np.array(result.salient_tokens_all_heads_all_layers_dicts)
    # Save the salient tokens as a numpy array
    np.save(os.path.join(folder_path, f"salient_tokens_dict.npy"), salient_tokens_all_heads_all_layers_dicts_array)
    # # Uncomment to save the salient tokens as a json file
    # with open(os.path.join(folder_path, f"salient_tokens_dict.json"), 'w', encoding="utf-8") as f:
    #     json.dump(result.salient_tokens_all_heads_all_layers_dicts, f, indent=4)
    
    # Save the thoughts token map lengths (as numpy)
    np.save(os.path.join(folder_path, f"thoughts_token_map_lengths.npy"), result.thoughts_token_map_lengths)

    # initialize a dictionary of the means for all context windows:
    means_all_context_windows = {context_window: None for context_window in args.context_windows}
    stds_all_context_windows = {context_window: None for context_window in args.context_windows}
    means_top_k_all_context_windows = {context_window: None for context_window in args.context_windows}
    stds_top_k_all_context_windows = {context_window: None for context_window in args.context_windows}
    
    for context_window in args.context_windows:
        # Create a subfolder for the current context window
        os.makedirs(os.path.join(folder_path, f"win_{context_window}"), exist_ok=True)
        # Save the thought interaction matrix mean attention scores
        np.save(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores.npy"), \
            result.dict_result_all_context_windows[context_window]['mean_all_scores'])
        np.save(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores.npy"), \
            result.dict_result_all_context_windows[context_window]['mean_top_k_scores'])
        # save result.dict_result_all_context_windows[context_window]['mean_top_k_indices_lists'] as numpy array
        thought_interaction_mat_mean_topk_indices_array = np.array(result.dict_result_all_context_windows[context_window]['mean_top_k_indices_lists'])
        np.save(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_indices.npy"), thought_interaction_mat_mean_topk_indices_array)
        # # Uncomment to save in json os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_indices.json")
        # with open(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_indices.json"), 'w', encoding="utf-8") as f:
        #     json.dump(result.dict_result_all_context_windows[context_window]['mean_top_k_indices_lists'], f, indent=4)

        means_all_context_windows[context_window], stds_all_context_windows[context_window] = process_mean_std(data=result.dict_result_all_context_windows[context_window]['mean_all_scores'])
        means_top_k_all_context_windows[context_window], stds_top_k_all_context_windows[context_window] = process_mean_std(data=result.dict_result_all_context_windows[context_window]['mean_top_k_scores'])

    # set the vmax ahead to normalize the heatmaps across all context windows and all heads
    vmax_means = max([np.max(means_all_context_windows[context_window]) for context_window in args.context_windows])
    vmax_stds = max([np.max(stds_all_context_windows[context_window]) for context_window in args.context_windows])
    vmax_means_top_k = max([np.max(means_top_k_all_context_windows[context_window]) for context_window in args.context_windows])
    vmax_stds_top_k = max([np.max(stds_top_k_all_context_windows[context_window]) for context_window in args.context_windows])
        
    for context_window in args.context_windows:
        title_mean = f"Thought Interactions TopK Attention Scores Mean along heads -\n{args.model_name} {args.dataset_name} {i} Context Window {context_window}"
        title_std = f"Thought Interactions TopK Attention Scores Std along heads -\n{args.model_name} {args.dataset_name} {i} Context Window {context_window}"
        title_mean_topk = f"Thought Interactions Attention Scores Mean along heads -\n{args.model_name} {args.dataset_name} {i} Context Window {context_window}"
        title_std_topk = f"Thought Interactions Attention Scores Std along heads -\n{args.model_name} {args.dataset_name} {i} Context Window {context_window}"

        # Set vmax for visualization as the max values over all context windows

        visualize_thoughts_interactions(means_top_k_all_context_windows[context_window], title_mean_topk,
                                        os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores_mean_along_heads.png"),
                                        vmax=vmax_means_top_k)
        visualize_thoughts_interactions(stds_top_k_all_context_windows[context_window], title_std_topk, 
                                        os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores_std_along_heads.png"),
                                        vmax=vmax_stds_top_k)
        visualize_thoughts_interactions(means_all_context_windows[context_window], title_mean,
                                        os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores_mean_along_heads.png"),
                                        vmax=vmax_means)
        visualize_thoughts_interactions(stds_all_context_windows[context_window], title_std,
                                        os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores_std_along_heads.png"),
                                        vmax=vmax_stds)
