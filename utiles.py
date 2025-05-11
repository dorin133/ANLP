from transformers import AutoTokenizer

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
