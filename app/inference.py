"""Inference for Legal LoRA Tagger."""

import re

import torch
import torch.nn.functional as F

from app.data_prep import CLAUSE_LABELS
from app.model import get_special_tokens, get_label_token_ids
from app.utils import get_device, get_logger

logger = get_logger(__name__)

LABEL_TOKENS = [f"<{l}>" for l in CLAUSE_LABELS]


# ---------------------------------------------------------------------------
# Single-clause prediction
# ---------------------------------------------------------------------------

def predict_clause(text: str, model, tokenizer, device=None, max_new_tokens: int = 10) -> dict:
    """
    Predict the label for a single legal clause.

    Strategy:
    1. Encode the input text.
    2. Run a forward pass and collect logits at the last token position.
    3. Score only the label special tokens.
    4. Apply softmax over those scores to get per-label probabilities.
    5. Return the highest-scoring label.

    Returns dict with keys: label, token, confidence.
    """
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)

    label_token_ids = get_label_token_ids(tokenizer)

    prompt = text.strip()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        last_logits = logits[0, -1, :]  # logits for next-token prediction

        label_scores = {
            label: last_logits[tid].item()
            for label, tid in label_token_ids.items()
        }

        if not label_scores:
            return {"label": "Unknown", "token": None, "confidence": None}

        ids_list = list(label_scores.keys())
        scores_tensor = torch.tensor([label_scores[l] for l in ids_list])
        probs = F.softmax(scores_tensor, dim=0)
        best_idx = probs.argmax().item()
        best_label = ids_list[best_idx]
        confidence = probs[best_idx].item()

    return {
        "label": best_label,
        "token": f"<{best_label}>",
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Document-level prediction
# ---------------------------------------------------------------------------

def predict_document(clauses: list, model, tokenizer, device=None) -> list:
    """
    Predict labels for a list of clauses (a document).
    Returns list of dicts with text, label, confidence.
    """
    results = []
    for clause in clauses:
        result = predict_clause(clause, model, tokenizer, device=device)
        results.append(
            {
                "text": clause,
                "label": result["label"],
                "confidence": result["confidence"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_tagged_document(results: list) -> str:
    """Format prediction results as a human-readable tagged document."""
    lines = []
    for i, item in enumerate(results, 1):
        lines.append(f"CLAUSE {i}:")
        lines.append(item["text"])
        lines.append("")
        lines.append("PREDICTED LABEL:")
        lines.append(item["label"])
        lines.append("")
        if item.get("confidence") is not None:
            conf_pct = item["confidence"] * 100
            lines.append("CONFIDENCE:")
            lines.append(f"{conf_pct:.1f}%")
            lines.append("")
        lines.append("-" * 60)
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------

def split_document_into_clauses(text: str) -> list:
    """
    Split a document text into clauses.
    Heuristic: split on double newlines or numbered/lettered sections.
    """
    clauses = re.split(r"\n\s*\n", text.strip())
    clauses = [c.strip() for c in clauses if c.strip()]
    return clauses


# ---------------------------------------------------------------------------
# Load model for inference
# ---------------------------------------------------------------------------

def load_model_for_inference(model_dir: str, adapter_path: str = None, device=None):
    """
    Load a trained model for inference.
    - Loads base tokenizer + model
    - Optionally loads LoRA adapter
    Returns (model, tokenizer).
    """
    import os
    from app.model import load_tokenizer_and_model
    from app.lora import GPT2LoRAModel, load_lora_model

    if device is None:
        device = get_device()

    tokenizer, base_model = load_tokenizer_and_model()

    if adapter_path and os.path.exists(adapter_path):
        model = load_lora_model(base_model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")
    else:
        model = GPT2LoRAModel(base_model)
        logger.warning("No adapter found — using untrained base model")

    return model, tokenizer
