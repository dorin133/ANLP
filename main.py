import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
from datasets import concatenate_datasets
import numpy as np
from hypothesis_functions import hypothesis_run
import matplotlib.pyplot as plt

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
                                split="test[:10]", 
                                trust_remote_code=True
                            ),
                            "config": "2-shot"
                        },
        "aime2024": {
                        "dataset": concatenate_datasets(
                            [
                                load_dataset
                                (
                                    "Maxwell-Jia/AIME_2024", 
                                    split="train[:10]",
                                    trust_remote_code=True
                                )
                            ]
                        ).shuffle(seed=42),
                        "config": "2-shot"
                    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("logs", exist_ok=True)

# Attention hook function
attention_scores = {}

# Prepare Chain-of-Thought prompt
def prepare_prompt(example, dataset_name, shot_examples=None):
    if dataset_name == "gsm8k":
        return f"Q: {example['question']}\nLet's think step by step.\n"
    elif ("math" in dataset_name):
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['problem']}\nFull Solution: {ex['solution']}\n" 
                                    for ex in shot_examples]
                                )
        return incontext + f"\nQuestion: {example['problem']}\nFull Solution:"
    elif "aime" in dataset_name:
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['Problem']}\nFull Solution: {ex['Solution']}\n" 
                                    for ex in shot_examples]
                                )
        return incontext + f"\nQuestion: {example['Problem']}\nFull Solution:"
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
                                                # trust_remote_code=True,
                                                torch_dtype="auto",
                                                # device_map="auto",
                                                # low_cpu_mem_usage=True,
                                                # attn_implementation="flash_attention_2" 
                                            ).to('cuda:0')
    model.eval()

    for dataset_name, dataset in datasets.items():
        print(f"Evaluating on {dataset_name}...")
        config = dataset.get("config", None)
        
        for i, example in enumerate(dataset['dataset']):
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
                output = model.generate(**inputs, 
                                max_new_tokens=max(max(1024, model.config.max_position_embeddings-inputs['input_ids'].shape[1]),0),
                            )
            
            folder_path = f"logs/{model_name}_{dataset_name}_{i}_2shot"
            os.makedirs(folder_path, exist_ok=True)
            # Save the input prompt:
            full_sequence_path = f"{folder_path}/full_sequence.txt"
            answers_path = f"{folder_path}/generated_answer.txt"
            log_filename_thought_interactions = f"{folder_path}/thought_interactions.txt"
            log_filename_salient_thoughts = f"{folder_path}/salient_thoughts.txt"

            # Save the entire sequence:
            with open(full_sequence_path, 'w', encoding="utf-8") as f:
                full_sequence = tokenizer.decode(output[0], skip_special_tokens=True)
                f.write(full_sequence)  
                          
            # Save the model's generated answer:
            with open(answers_path, 'w', encoding="utf-8") as f:
                generated_ids = [
                            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
                        ]
                answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                f.write(answer)
                
            all_interactions, thought_interaction_matrix_mean_attn_scores = hypothesis_run(
                        model,
                        tokenizer,
                        sequence_path=full_sequence_path,
                        log_filename_thought_interactions=log_filename_thought_interactions,
                        log_filename_salient_thoughts=log_filename_salient_thoughts
                    )
            for layer in thought_interaction_matrix_mean_attn_scores.keys():
                for head in thought_interaction_matrix_mean_attn_scores[layer].keys():
                    plt.figure(figsize=(10, 6))
                    plt.subplot(1, 1, 1)
                    plt.imshow(thought_interaction_matrix_mean_attn_scores[layer][head], cmap='viridis')
                    plt.colorbar(label='Mean Attention Score')
                    plt.title(f'Thought Interactions - Layer {layer}, Head {head}')
                    plt.ylabel('Current Thought Index')
                    plt.xlabel('Previous Thought Index')
                    
                    # Use actual thought indices as ticks
                    plt.xticks(range(thought_interaction_matrix_mean_attn_scores[layer][head].shape[0]), range(thought_interaction_matrix_mean_attn_scores[layer][head].shape[0]))
                    plt.yticks(range(thought_interaction_matrix_mean_attn_scores[layer][head].shape[0]), range(thought_interaction_matrix_mean_attn_scores[layer][head].shape[0]))
                    # plt.tight_layout()

                    plt.savefig(f"logs/{model_name}_{dataset_name}_{i}_2shot/thought_interaction_matrix_plot_layer{layer}_head{head}.png")
                    plt.show()
                    plt.close()
