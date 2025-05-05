import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
from datasets import concatenate_datasets
import numpy as np

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
                                split="test[:5]", 
                                trust_remote_code=True
                            ),
                            "config": "2-shot"
                        },
        # "math-all": {
        #                 "dataset": concatenate_datasets(
        #                     [
        #                         load_dataset
        #                         (
        #                             "EleutherAI/hendrycks_math", 
        #                             hendrycks_math_names[i], 
        #                             split="test[:5]", 
        #                             trust_remote_code=True
        #                         ) 
        #                         for i in range(len(hendrycks_math_names))
        #                     ]
        #                 ).shuffle(seed=42),
        #                 "config": "2-shot"
        #             },
    # "aime2024": load_dataset("Maxwell-Jia/AIME_2024")  
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("attention_matrices", exist_ok=True)
os.makedirs("full_sequences", exist_ok=True)
os.makedirs("generated_answers", exist_ok=True)

# Attention hook function
attention_scores = {}

def save_attention_hook(layer_id):
    def hook(module, input, output):
        # output[2] contains the attention weights in transformers
        # output: (attn_output, attn_weights) or (attn_output, present, attn_weights)
        if isinstance(output, tuple) and len(output) > 2:
            attn = output[2]
        elif isinstance(output, tuple) and len(output) == 2:
            attn = output[1]
        if attn is not None:
            attention_scores[layer_id] = attn.detach().cpu()
    return hook

# Prepare Chain-of-Thought prompt
def prepare_prompt(example, dataset_name, shot_examples=None):
    if dataset_name == "gsm8k":
        return f"Q: {example['question']}\nLet's think step by step.\n"
    elif "math" in dataset_name:
        if shot_examples:
            incontext = "\n".join(
                                    [f"Question: {ex['problem']}\nFull Solution: {ex['solution']}\n" 
                                    for ex in shot_examples]
                                )
        return incontext + f"\nQuestion: {example['problem']}\nFull Solution:"
    elif dataset_name == "aime2024":
        return f"Problem: {example['question']}\nStep-by-step solution:"
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

    # # Register hooks on all attention layers
    # hooks = []
    # for i, layer in enumerate(model.model.layers):  # adjust to model internals
    #     h = layer.self_attn.register_forward_hook(save_attention_hook(i))
    #     hooks.append(h)

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
            with torch.no_grad():
                forward_output = model(output, output_attentions=True)
            
            output_path = f"attention_matrices/{model_name}_{dataset_name}_{i}_2shot.pt"
            full_sequence_path = f"full_sequences/{model_name}_{dataset_name}_{i}_2shot.txt"
            answers_path = f"generated_answers/{model_name}_{dataset_name}_{i}_2shot.txt"
            
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
                
            # Save attention scores
            attention_scores = forward_output["attentions"]
            with open(output_path, 'wb') as f:
                torch.save(attention_scores, f)
            print(f"Saved: {full_sequence_path}")
            
            # Clear attention scores for the next example              
            del attention_scores
            attention_scores = {} 


