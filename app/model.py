"""Model loading and tokenizer setup for Legal LoRA Tagger."""

from app.data_prep import CLAUSE_LABELS

MODEL_NAME = "distilgpt2"  # Default; can override with "gpt2"


def get_special_tokens():
    """Return list of special tokens for each clause label."""
    return [f"<{label}>" for label in CLAUSE_LABELS]


def load_tokenizer_and_model(model_name: str = MODEL_NAME, add_special_tokens: bool = True):
    """
    Load GPT-2/DistilGPT-2 tokenizer and model.
    - Adds one special token per clause label
    - Sets pad_token = eos_token
    - Resizes model embeddings
    Returns: (tokenizer, model)
    """
    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    if add_special_tokens:
        special_tokens = get_special_tokens()
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        # Allow gradients on new embedding rows
        model.transformer.wte.weight.requires_grad = True

    return tokenizer, model


def get_label_token_ids(tokenizer) -> dict:
    """Return dict mapping label -> token_id for each special token."""
    result = {}
    for label in CLAUSE_LABELS:
        token = f"<{label}>"
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            result[label] = ids[0]
    return result
