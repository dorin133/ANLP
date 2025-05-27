import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import seaborn as sns
import json
from datasets import concatenate_datasets
import numpy as np
from hypothesis_functions import hypothesis_run, init_output_dir
from utils.hypothesis_utils import process_and_save_mean_std
import matplotlib.pyplot as plt
from utils.visualize_salient_thoughts import visualize_thoughts_interactions, visualize_salient_thoughts
import copy

# parse arguments from the .sh script passed to main.py
import argparse
parser = argparse.ArgumentParser(description="Run the model with specified parameters.")
parser.add_argument(
    "--max_new_tokens", 
    type=int, 
    default=1024, 
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
    default=10,
    help="Number of samples to generate per task."
)
parser.add_argument(
    "--overwrite_output_dir",
    type=bool,
    default=False,
    help="Overwrite the output directory if it exists."
)
# parse argument of the context windows list
parser.add_argument(
    "--context_windows",
    type=str,
    default="[-1, 15]",
    help="List of context windows to use."
)
args = parser.parse_args()
# Convert the context windows string to a list of integers
args.context_windows = eval(args.context_windows)

# Configuration
models = {
    "Qwen2.5-Math": "Qwen/Qwen2.5-Math-7B-Instruct",
    # "DeepSeek-R1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
}
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
        "math-algebra": {
                            "dataset": load_dataset
                            (
                                "EleutherAI/hendrycks_math", 
                                hendrycks_math_names[0], 
                                split="test[:100]", 
                                trust_remote_code=True
                            ).shuffle(seed=42).select(np.arange(args.num_samples_per_task)),
                            "config": "2-shot"
                        },
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
                        ).shuffle(seed=7).select(np.arange(args.num_samples_per_task)),
                        "config": "2-shot"
                    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if overwrite_output_dir is True, initialize the output directory to be empty
if args.overwrite_output_dir:
    init_output_dir(args.output_dir, args.overwrite_output_dir)

# Attention hook function
attention_scores = {}

# Prepare Chain-of-Thought prompt
def prepare_prompt(example, dataset_name, shot_examples=None):
    CoT_instruct = "Let's think step by step. Each thought should be separated by exactly two newline characters r'\\n\\n'.\n"
    if dataset_name == "gsm8k":
        return f"Q: {example['question']}\n{CoT_instruct}"
    elif ("math" in dataset_name):
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['problem']}\nFull Solution: {ex['solution']}\n" 
                                    for ex in shot_examples]
                                )
        return CoT_instruct + incontext + f"\nQuestion: {example['problem']}\nFull Solution:"
    elif "aime" in dataset_name:
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['Problem']}\nFull Solution: {ex['Solution']}\n" 
                                    for ex in shot_examples]
                                )
        return CoT_instruct + incontext + f"\nQuestion: {example['Problem']}\nFull Solution:"
    else:
        return example['text']

# === Main evaluation loop ===
for model_name, model_path in models.items():
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                                                model_path, 
                                                # output_attentions=True, 
                                                # return_dict_in_generate=True,
                                                trust_remote_code=True,
                                                torch_dtype="auto",
                                                device_map="auto",
                                                # low_cpu_mem_usage=True,
                                                attn_implementation="eager" 
                                            )
    model.eval()

    for dataset_name, dataset in datasets.items():
        print(f"Evaluating on {dataset_name}...")
        config = dataset.get("config", None)
        
        for i, example in enumerate(dataset['dataset']):
            
            folder_path = f"{args.output_dir}/{model_name}_{dataset_name}_{i}_{config}"
            
            # check if os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores.npy") exists
            unseen_context_windows = copy.deepcopy(args.context_windows)
            for context_window in args.context_windows:
                if not os.path.exists(os.path.join(folder_path, f"win_{context_window}")):
                    os.makedirs(os.path.join(folder_path, f"win_{context_window}"), exist_ok=True)
                    
                if os.path.exists(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores.npy")) and \
                    os.path.exists(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores.npy")):
                    
                    if not args.overwrite_output_dir:
                        # remove the context window folder if it exists and we don't want to overwrite it
                        unseen_context_windows.remove(context_window)
                        
            if unseen_context_windows == []:
                # if the list is empty, skip this example
                print(f"Skipping example {i} for {model_name} on {dataset_name} because all context windows have been processed.")
                continue
            
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
            prompt = prepare_prompt(example, dataset_name, shot_examples)
            messages = [
                        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                        {"role": "user", "content": prompt}
                    ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                if os.path.exists(os.path.join(folder_path, "full_sequence.txt")) and not args.overwrite_output_dir:
                    with open(os.path.join(folder_path, "full_sequence.txt"), 'r', encoding="utf-8") as f:
                        full_sequence = f.read()
                        output = tokenizer([full_sequence])['input_ids']
                else:
                    output = model.generate(**inputs, 
                                max_new_tokens=max(min(args.max_new_tokens, model.config.max_position_embeddings-inputs['input_ids'].shape[1]),0),
                            )

            # Save the input prompt:
            os.makedirs(folder_path, exist_ok=True)
            full_sequence_path = f"{folder_path}/full_sequence.txt"
            answers_path = f"{folder_path}/generated_answer.txt"
            log_filename_thought_interactions = f"{folder_path}/thought_interactions.txt"
            log_filename_salient_thoughts = f"{folder_path}/salient_thoughts.txt"

            if not os.path.exists(os.path.join(folder_path, "full_sequence.txt")) and not args.overwrite_output_dir:
                # Save the entire sequence:
                with open(full_sequence_path, 'w', encoding="utf-8") as f:
                    full_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
                    f.write(full_sequence)  

            if not os.path.exists(os.path.join(folder_path, "generated_answer.txt")) and not args.overwrite_output_dir:
                # Save the model's generated answer:
                with open(answers_path, 'w', encoding="utf-8") as f:
                    generated_ids = [
                                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
                            ]
                    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    f.write(answer)

            if unseen_context_windows != []: # if the list is not empty
                # run the hypothesis function on the unseen context windows
                result = hypothesis_run(
                    model,
                    tokenizer,
                    sequence_path=full_sequence_path,
                    log_filename_thought_interactions=log_filename_thought_interactions,
                    log_filename_salient_thoughts=log_filename_salient_thoughts,
                    context_windows=unseen_context_windows,
                )

                # Visualize the salient thoughts for the current context window
                visualize_salient_thoughts(data=result.salient_thoughts_all_heads_all_layers_array, \
                                            title=f"Salient Thoughts - {model_name} {dataset_name} {i}",
                                            output_dir=folder_path)
                    
                # Save the salient thoughts (this result is independent of the context windows)
                np.save(os.path.join(folder_path, f"salient_thoughts.npy"), result.salient_thoughts_all_heads_all_layers_array)
                
                for context_window in unseen_context_windows:
                    # Save the thought interaction matrix mean attention scores
                    np.save(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores.npy"), \
                        result.dict_result_all_context_windows[context_window]['mean_all_scores'])
                    np.save(os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores.npy"), \
                        result.dict_result_all_context_windows[context_window]['mean_top_k_scores'])
                    
                    means, stds = process_and_save_mean_std(data=result.dict_result_all_context_windows[context_window]['mean_all_scores'])
                    means_topk, stds_topk = process_and_save_mean_std(data=result.dict_result_all_context_windows[context_window]['mean_top_k_scores'])
                    
                    title_mean = f"Thought Interactions TopK Attention Scores Mean along heads -\n{model_name} {dataset_name} {i} Context Window {context_window}"
                    title_std = f"Thought Interactions TopK Attention Scores Std along heads -\n{model_name} {dataset_name} {i} Context Window {context_window}"
                    title_mean_topk = f"Thought Interactions Attention Scores Mean along heads -\n{model_name} {dataset_name} {i} Context Window {context_window}"
                    title_std_topk = f"Thought Interactions Attention Scores Std along heads -\n{model_name} {dataset_name} {i} Context Window {context_window}"

                    visualize_thoughts_interactions(means_topk, title_mean_topk, 
                                                    os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores_mean_along_heads.png"))
                    visualize_thoughts_interactions(stds_topk, title_std_topk, 
                                                    os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_topk_attn_scores_std_along_heads.png"))
                    visualize_thoughts_interactions(means, title_mean,
                                                    os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores_mean_along_heads.png"))
                    visualize_thoughts_interactions(stds, title_std,
                                                    os.path.join(folder_path, f"win_{context_window}", f"thought_interaction_mat_mean_attn_scores_std_along_heads.png"))
