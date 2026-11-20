[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/d9vsyjrnms/badge.svg)](https://ask.thili.ai/b/d9vsyjrnms)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/zgqr45ere6/badge.svg)](https://ask.thili.ai/b/zgqr45ere6)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/fyvrjpp46y/badge.svg)](https://ask.thili.ai/b/fyvrjpp46y)

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ng2qks3qxx/badge.svg)](https://ask.thili.ai/b/ng2qks3qxx)
[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/xmc53n5bmt/badge.svg)](https://ask.thili.ai/b/xmc53n5bmt)

# Legal Document LoRA Tagger

![CI](https://github.com/your-org/legal-document-lora-tagger/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Model](https://img.shields.io/badge/model-DistilGPT--2-orange)
![LoRA](https://img.shields.io/badge/adapter-LoRA-purple)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [System Architecture](#4-system-architecture)
5. [Technologies Used](#5-technologies-used)
6. [Dataset Preparation](#6-dataset-preparation)
7. [GPT-2 Tokenizer Extension](#7-gpt-2-tokenizer-extension)
8. [LoRA Implementation Details](#8-lora-implementation-details)
9. [Training Instructions](#9-training-instructions)
10. [Inference Instructions](#10-inference-instructions)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Example Input / Output](#12-example-input--output)
13. [Installation Instructions](#13-installation-instructions)
14. [Full CLI Usage](#14-full-cli-usage)
15. [Limitations and Future Improvements](#15-limitations-and-future-improvements)

---

## 1. Project Overview

**Legal Document LoRA Tagger** is an end-to-end machine-learning pipeline that automatically classifies individual clauses inside legal contracts. Given a paragraph of legal text, the model outputs a structured label â€” such as `Governing_Law`, `Termination`, `Indemnification`, or one of 17 other clause types â€” enabling downstream automation of contract review, risk analysis, and compliance checking.

The system is built on top of **DistilGPT-2** (or GPT-2), a transformer language model, fine-tuned using **LoRA (Low-Rank Adaptation)** â€” a parameter-efficient fine-tuning technique that trains less than 1% of model parameters while delivering performance competitive with full fine-tuning.

### Key highlights

| Feature | Detail |
|---|---|
| Model | DistilGPT-2 (82M params) or GPT-2 (117M params) |
| Fine-tuning method | LoRA (custom implementation, no peft) |
| Trainable params | ~0.1â€“0.5% of total |
| Number of classes | 20 legal clause types |
| Dataset | Built-in sample data + optional LEDGAR (via HuggingFace) |
| CLI | Full argparse CLI with 7 commands |
| Testing | pytest with graceful skip when torch is absent |
| CI | GitHub Actions (Python 3.10 & 3.11) |

---

## 2. Problem Statement

### The Legal NLP Challenge

Legal contracts are dense, long documents written in formal legal prose. A single contract may contain dozens of distinct clause types â€” governing law, termination conditions, payment schedules, confidentiality requirements â€” often interleaved without clear section boundaries.

**Manual review is expensive:**
- A junior associate may spend 4â€“8 hours reviewing a single complex contract.
- Large enterprises process thousands of contracts per year.
- Errors in contract interpretation lead to real financial and legal risk.

**Automated approaches face several challenges:**

1. **Domain vocabulary**: Legal language is highly specialized. Standard NLP models pre-trained on general text may miss subtle distinctions (e.g., "assignment" in a legal context means transfer of contractual rights, not a task).

2. **Class imbalance**: Real contracts may have 10 termination clauses but only 1 force-majeure clause.

3. **Data scarcity**: Annotated legal datasets are expensive to build; publicly available datasets like LEDGAR are limited.

4. **Interpretability requirements**: Legal professionals need to understand *why* a clause was classified a certain way.

5. **Deployment constraints**: Models must run efficiently in enterprise environments, potentially without GPUs.

This project addresses these challenges by combining the linguistic understanding of GPT-2 with the efficiency of LoRA fine-tuning and a carefully designed special-token tagging strategy.

---

## 3. Proposed Solution

### GPT-2 + LoRA: Parameter-Efficient Fine-Tuning for Legal NLP

#### Why GPT-2?

GPT-2 is a transformer decoder model pre-trained on 40GB of internet text (WebText). Despite being a general-purpose language model, it has strong zero-shot and few-shot capabilities for text classification when framed as a text generation problem. We use it in a **generative classification** mode:

> *Given clause text, predict the most likely label token as the next token.*

This approach:
- Leverages GPT-2's pre-trained linguistic knowledge
- Avoids the need for a separate classification head
- Makes adding new label classes easy (just add new special tokens)

#### Why LoRA?

Full fine-tuning of GPT-2 (117M parameters) on a small legal dataset risks **catastrophic forgetting** and **overfitting**. LoRA addresses this by:

1. **Freezing all base model weights** â€” the original knowledge is preserved
2. **Injecting small trainable rank-decomposition matrices** into attention layers
3. **Training only ~0.1â€“0.5% of parameters** â€” faster, cheaper, less prone to overfitting
4. **Saving only the adapter** â€” the base model never changes, so adapters can be swapped

#### Why custom LoRA (no peft)?

This project implements LoRA from scratch using PyTorch's forward hook mechanism. This approach:
- Works transparently with GPT-2's internal `Conv1D` layers (which differ from standard `nn.Linear`)
- Requires no additional dependencies
- Is educational â€” demonstrates exactly how LoRA works at the mathematical level

---

## 4. System Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                      Legal Document LoRA Tagger                      â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

  Input Text (legal clause)
        â”‚
        â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚  Tokenizer  â”‚  GPT2Tokenizer + 20 special label tokens
  â”‚  (extended) â”‚  <Governing_Law>, <Termination>, ...
  â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜
         â”‚  token IDs
         â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚              DistilGPT-2 Base Model               â”‚
  â”‚  (all weights FROZEN â€” requires_grad=False)       â”‚
  â”‚                                                   â”‚
  â”‚  Embedding â†’ [Block 0] â†’ [Block 1] â†’ ... â†’       â”‚
  â”‚  [Block 5] â†’ LayerNorm â†’ LM Head                 â”‚
  â”‚                                                   â”‚
  â”‚   Each Block:                                     â”‚
  â”‚   â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”  â”‚
  â”‚   â”‚  LayerNorm                                â”‚  â”‚
  â”‚   â”‚  c_attn (Q,K,V fused) â”€â”€â”€ LoRA Hook â”€â”€â–º  â”‚  â”‚
  â”‚   â”‚                       + LoRA delta        â”‚  â”‚
  â”‚   â”‚  Attention + Softmax                      â”‚  â”‚
  â”‚   â”‚  c_proj â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ LoRA Hook â”€â”€â”€â”€â”€â”€â–º  â”‚  â”‚
  â”‚   â”‚                       + LoRA delta        â”‚  â”‚
  â”‚   â”‚  LayerNorm + MLP                          â”‚  â”‚
  â”‚   â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜  â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                             â”‚  logits (vocab_size)
                             â–¼
  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
  â”‚            Label Token Scoring                   â”‚
  â”‚                                                  â”‚
  â”‚  Extract logits for the 20 special label tokens  â”‚
  â”‚  Apply softmax â†’ probabilities                   â”‚
  â”‚  Return argmax label + confidence score          â”‚
  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
         â”‚
         â–¼
  Output: { label: "Governing_Law", confidence: 0.87 }
```

### LoRA Hook Mechanism

```
Forward pass through c_attn:

  x â”€â”€â–º c_attn (frozen) â”€â”€â–º base_output
  â”‚
  â””â”€â”€â–º LoRALayer
         lora_A: (r, in_features)   kaiming_uniform init
         lora_B: (out_features, r)  zero init
         delta = dropout(x) @ A.T @ B.T * (alpha/r)
         â”€â”€â–º delta
  
  final_output = base_output + delta
```

---

## 5. Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core runtime |
| PyTorch | â‰¥2.0.0 | Deep learning framework, autograd |
| Transformers (HuggingFace) | â‰¥4.35.0 | GPT-2 model and tokenizer |
| Datasets (HuggingFace) | â‰¥2.14.0 | Optional LEDGAR dataset loading |
| scikit-learn | â‰¥1.3.0 | Evaluation metrics (accuracy, F1, etc.) |
| NumPy | â‰¥1.24.0 | Numerical utilities, random seed |
| tqdm | â‰¥4.65.0 | Progress bars |
| python-dotenv | â‰¥1.0.0 | Environment configuration |
| pytest | â‰¥7.0.0 | Test framework |
| GitHub Actions | â€” | CI/CD pipeline |

### Why these choices?

- **PyTorch**: Industry-standard, flexible, excellent debugging support via eager mode
- **DistilGPT-2**: 40% smaller than GPT-2, 60% faster at inference, retains 97% of GPT-2 performance â€” ideal for resource-constrained environments
- **HuggingFace Transformers**: De-facto standard for pre-trained NLP models; seamless model loading with `from_pretrained`
- **Custom LoRA**: Educational, no extra dependencies, full control over hook placement

---

## 6. Dataset Preparation

### Built-in Sample Data

The project ships with a curated set of **at least 3 expert-written example clauses per label** (20 labels Ã— 5 examples = 100 examples). These are realistic legal texts covering all 20 clause types.

Split strategy: **stratified 70/15/15** (train/val/test) per class, ensuring:
- Every class appears in every split
- No text appears in more than one split (no data leakage)
- Deterministic with `seed=42`

### LEDGAR Dataset (Optional)

[LEDGAR](https://huggingface.co/datasets/lex_glue) is a large-scale real-world legal clause dataset from the LexGLUE benchmark, containing ~60,000 contract provisions from SEC filings, annotated with 100 provision types.

To use LEDGAR:

```bash
python -m app.cli train --ledgar
```

The pipeline:
1. Attempts to load `lex_glue/ledgar` via HuggingFace `datasets`
2. Maps LEDGAR's label names to our 20 `CLAUSE_LABELS` where possible
3. Falls back to built-in sample data with a clear warning if LEDGAR is unavailable

```
load_dataset_for_training(use_ledgar=True)
    â”œâ”€â”€ try_load_ledgar()
    â”‚       â”œâ”€â”€ load_dataset("lex_glue", "ledgar")   # HuggingFace
    â”‚       â”œâ”€â”€ map labels â†’ CLAUSE_LABELS
    â”‚       â””â”€â”€ return (train, val, test) or None
    â””â”€â”€ fallback: get_sample_dataset()               # always works
```

### LegalClauseDataset

The `LegalClauseDataset` (a `torch.utils.data.Dataset`) tokenizes each clause as:

```
"{clause text} <{label}>"
```

This **generative framing** trains GPT-2 to predict the label token as the natural continuation of a legal clause. The model learns which label token is most likely given the clause context.

---

## 7. GPT-2 Tokenizer Extension

GPT-2's tokenizer uses Byte-Pair Encoding (BPE) and has no built-in concept of legal clause labels. We extend it by adding **20 special tokens** â€” one per clause label.

```python
special_tokens = ["<Governing_Law>", "<Termination>", "<Confidentiality>", ...]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))
```

**Why special tokens instead of regular text?**

| Approach | Pros | Cons |
|---|---|---|
| Regular text (e.g., `"governing law"`) | No vocab change | Tokenized into multiple sub-tokens; ambiguous |
| Special tokens (e.g., `<Governing_Law>`) | Single token; unambiguous; no BPE splitting | Requires vocab resize |

By adding special tokens:
1. Each label maps to exactly **one token ID** â€” no multi-token ambiguity
2. The model can be evaluated by simply checking the probability of that one token
3. New embedding rows are initialized at random and become the learnable "label anchors"

```python
# After adding special tokens:
tokenizer.encode("<Governing_Law>", add_special_tokens=False)
# â†’ [50257]  (a single new token ID)
```

**Embedding resize**: After `resize_token_embeddings`, the new rows are set to `requires_grad=True` so they are updated during training alongside the LoRA parameters.

---

## 8. LoRA Implementation Details

### Mathematical Foundation

Standard fine-tuning modifies a weight matrix **W** directly:

```
W_new = W + Î”W
```

LoRA constrains **Î”W** to have low rank **r â‰ª min(d_in, d_out)**:

```
Î”W = B @ A     where A âˆˆ â„^{r Ã— d_in},  B âˆˆ â„^{d_out Ã— r}
```

The forward pass becomes:

```
h = W x + (B @ A) x Â· (Î± / r)
  = W x + B (A x) Â· scaling
```

where **Î±** is a scaling hyperparameter that controls the magnitude of the adaptation.

**Initialization**:
- **A** is initialized with `kaiming_uniform` (like default `nn.Linear`) â€” provides diverse gradients from the first step
- **B** is initialized to **zero** â€” so at the start of training, `Î”W = 0` and the model is identical to the base model

### Implementation in This Project

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.1):
        super().__init__()
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (..., in_features) â†’ output: (..., out_features)
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### Hook-Based Injection

GPT-2 uses `Conv1D` (not `nn.Linear`) for its attention projections. `Conv1D` has weight shape `(in, out)` â€” the transpose of `nn.Linear`. Replacing these layers directly is fragile.

Instead, we use **PyTorch forward hooks**: a function called automatically after each targeted module's forward pass that adds the LoRA delta to the output:

```python
def make_hook(lora_layer):
    def hook(module, input, output):
        x = input[0]
        delta = lora_layer(x)
        return output + delta
    return hook

module.register_forward_hook(make_hook(lora))
```

**Target modules**: `c_attn` (fused Q, K, V projection, shape 768â†’2304) and `c_proj` (output projection, shape 768â†’768).

### Hyperparameters

| Hyperparameter | Default | Meaning |
|---|---|---|
| `r` | 4 | LoRA rank â€” controls expressiveness vs. parameter count |
| `alpha` | 16 | Scaling factor â€” controls update magnitude |
| `dropout` | 0.1 | Dropout on input to LoRA branch |
| `lr` | 3e-4 | AdamW learning rate |
| `epochs` | 3 | Training epochs |
| `batch_size` | 4 | Batch size |

**Parameter count** for `distilgpt2` with `r=4`:
- Base model: ~82M params (all frozen)
- LoRA params: ~12 layers Ã— 2 targets Ã— 2 matrices Ã— (r Ã— dim) â‰ˆ ~50K params
- Trainable %: < 0.1%

---

## 9. Training Instructions

### Prerequisites

Install all dependencies first (see [Section 13](#13-installation-instructions)).

### Basic training (built-in sample data)

```bash
python -m app.cli train
```

### Training with custom hyperparameters

```bash
python -m app.cli train \
    --lora-rank 8 \
    --lora-alpha 32 \
    --epochs 5 \
    --batch-size 8 \
    --lr 2e-4 \
    --output-dir ./my_checkpoints \
    --n-per-class 10
```

### Training with LEDGAR dataset

```bash
python -m app.cli train --ledgar --epochs 10 --batch-size 16
```

### Training flow

```
1. Load DistilGPT-2 tokenizer + model
2. Add 20 special label tokens â†’ resize embeddings
3. Wrap model in GPT2LoRAModel â†’ inject LoRA hooks
4. Freeze all base parameters
5. Build LegalClauseDataset (train, val)
6. AdamW optimizer on LoRA params only
7. For each epoch:
   a. Training loop: forward, loss.backward(), clip_grad_norm_, step
   b. Validation loop: eval mode, no_grad
   c. Save checkpoint if val_loss improved
8. Save final adapter to ./checkpoints/final_adapter/lora_adapter.pt
```

### Checkpoint structure

```
checkpoints/
â”œâ”€â”€ best_checkpoint/
â”‚   â””â”€â”€ lora_adapter.pt        â† best val loss checkpoint
â””â”€â”€ final_adapter/
    â””â”€â”€ lora_adapter.pt        â† final epoch checkpoint
```

Each `lora_adapter.pt` is a dict:
```python
{
    "lora_state_dict": {...},  # weights for all LoRA layers
    "r": 4,
    "alpha": 16,
    "config": {"r": 4, "alpha": 16}
}
```

---

## 10. Inference Instructions

### Predict a single clause

```bash
python -m app.cli predict \
    --text "This agreement shall be governed by the laws of the State of New York."
```

**Output:**
```
Input: This agreement shall be governed by the laws of the State of New York.

PREDICTED LABEL: Governing_Law
CONFIDENCE: 87.3%
```

### Tag a full document

```bash
python -m app.cli process-document \
    --input sample_contract.txt \
    --output tagged_contract.txt
```

### Batch processing

```bash
python -m app.cli batch \
    --input ./contracts/ \
    --output ./tagged_contracts/
```

### Programmatic inference

```python
from app.inference import load_model_for_inference, predict_clause

model, tokenizer = load_model_for_inference(
    model_dir="./checkpoints",
    adapter_path="./checkpoints/final_adapter/lora_adapter.pt"
)

result = predict_clause(
    "Payment shall be made within thirty days of receipt of invoice.",
    model,
    tokenizer
)
# â†’ {"label": "Payment_Terms", "token": "<Payment_Terms>", "confidence": 0.91}
```

### How inference works

1. Tokenize input clause text
2. Run forward pass through GPT2LoRAModel
3. Extract logits at the **last token position** (next-token prediction)
4. Score only the 20 label token IDs
5. Apply softmax to get per-label probabilities
6. Return `argmax` label + confidence

---

## 11. Evaluation Metrics

### Metrics computed

| Metric | Formula | Interpretation |
|---|---|---|
| **Accuracy** | correct / total | Overall fraction correctly classified |
| **Precision (weighted)** | Î£ (class_precision Ã— support) / total | How precise predictions are, weighted by class size |
| **Recall (weighted)** | Î£ (class_recall Ã— support) / total | How many true instances were found |
| **F1 (weighted)** | 2 Ã— P Ã— R / (P + R) | Harmonic mean of precision and recall |
| **Confusion matrix** | C[i][j] = predicted j when true i | Per-class breakdown of errors |

Metrics are computed using **scikit-learn** when available, with a manual fallback for accuracy-only when sklearn is absent.

### Baseline comparison

The evaluation includes a **majority-class baseline**: always predict the most common class in the test set. For balanced datasets (equal samples per class), this baseline achieves `1/20 = 5%` accuracy.

### Running evaluation

```bash
python -m app.cli evaluate --output-dir ./checkpoints
```

**Output example:**
```
=== Model Efficiency Report ===
Total parameters:     82,235,648
Trainable parameters: 49,152
Frozen parameters:    82,186,496
Trainable percentage: 0.0598%

Baseline (majority class 'Governing_Law'): 0.0500 accuracy

=== Evaluation Results ===
Accuracy:  0.4200
Precision: 0.4150
Recall:    0.4200
F1:        0.4100
Avg inference time: 45.3ms per clause
```

> **Note**: With only 3â€“5 examples per class and no LEDGAR data, accuracy will be modest. The goal of this project is to demonstrate the LoRA fine-tuning pipeline; production performance requires LEDGAR or a larger annotated corpus.

---

## 12. Example Input / Output

### Input: `sample_contract.txt`

```
The agreement shall be governed by and construed in accordance with the laws of India.

Either party may terminate this agreement upon thirty (30) days written notice to the other party.

Each party agrees to keep confidential all proprietary information disclosed by the other party
and shall not disclose such information to any third party.

The Company shall indemnify and hold harmless the Client from any claims, damages, or expenses
arising from the Company's breach of this agreement.

In no event shall either party be liable for any indirect, incidental, special, or consequential damages.
```

### Output: `tagged_contract.txt`

```
CLAUSE 1:
The agreement shall be governed by and construed in accordance with the laws of India.

PREDICTED LABEL:
Governing_Law

CONFIDENCE:
87.1%

------------------------------------------------------------

CLAUSE 2:
Either party may terminate this agreement upon thirty (30) days written notice to the other party.

PREDICTED LABEL:
Termination

CONFIDENCE:
81.5%

------------------------------------------------------------

CLAUSE 3:
Each party agrees to keep confidential all proprietary information...

PREDICTED LABEL:
Confidentiality

CONFIDENCE:
79.3%

------------------------------------------------------------
```

### Programmatic batch example

```python
from app.data_prep import load_dataset_for_training
from app.evaluate import evaluate_model, evaluate_baseline

# Load test data
_, _, test_data, _ = load_dataset_for_training(n_per_class=5)

# Baseline
bl = evaluate_baseline(test_data)
print(f"Baseline accuracy: {bl['accuracy']:.2%}")

# Model evaluation
metrics = evaluate_model(model, tokenizer, test_data)
print(f"Model accuracy:    {metrics['accuracy']:.2%}")
print(f"Model F1:          {metrics['f1']:.2%}")
```

---

## 13. Installation Instructions

### Prerequisites

- Python 3.10 or 3.11
- pip (bundled with Python)
- 4 GB RAM minimum (8 GB recommended)
- GPU optional (CUDA 11.8+ if using GPU)

### Step 1: Clone the repository

```bash
git clone https://github.com/your-org/legal-document-lora-tagger.git
cd legal-document-lora-tagger
```

### Step 2: Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install PyTorch

**CPU only (recommended for first run):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 11.8):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install project dependencies

```bash
pip install -r requirements.txt
```

### Step 5: (Optional) Install in editable mode

```bash
pip install -e .
```

After this, you can use the `lora-tagger` command directly:
```bash
lora-tagger hello
```

### Step 6: Configure environment

```bash
cp .env.example .env
# Edit .env to set HF_TOKEN if you plan to use LEDGAR
```

### Step 7: Verify installation

```bash
python -m app.cli hello
# â†’ Hello from Legal Document LoRA Tagger!

python -m pytest tests/ -v
# â†’ All pure-Python tests should pass
```

### Minimal install (no ML packages, for testing only)

If you only want to run the pure-Python tests without installing PyTorch or Transformers:

```bash
pip install pytest
python -m pytest tests/ -v
# torch-dependent tests will be skipped automatically
```

---

## 14. Full CLI Usage

The CLI is accessed as:

```bash
python -m app.cli <command> [options]
# or, if installed via setup.py:
lora-tagger <command> [options]
```

### `hello` â€” Print greeting

```bash
python -m app.cli hello
```

### `prepare-data` â€” Inspect dataset

```bash
python -m app.cli prepare-data [--ledgar] [--n-per-class N]
```

| Option | Default | Description |
|---|---|---|
| `--ledgar` | False | Try to load LEDGAR from HuggingFace |
| `--n-per-class` | 5 | Samples per class from built-in data |

### `train` â€” Train the LoRA model

```bash
python -m app.cli train [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--lora-rank` | 4 | LoRA rank r |
| `--lora-alpha` | 16 | LoRA scaling alpha |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 4 | Batch size |
| `--lr` | 3e-4 | AdamW learning rate |
| `--output-dir` | ./checkpoints | Where to save adapter |
| `--n-per-class` | 5 | Samples per class (sample data) |
| `--ledgar` | False | Use LEDGAR dataset |

### `evaluate` â€” Evaluate model

```bash
python -m app.cli evaluate [--output-dir DIR] [--n-per-class N]
```

### `predict` â€” Classify a single clause

```bash
python -m app.cli predict --text "TEXT" [--output-dir DIR]
```

### `process-document` â€” Tag a full document

```bash
python -m app.cli process-document --input FILE [--output FILE] [--output-dir DIR]
```

| Option | Description |
|---|---|
| `--input` | Path to input text file (required) |
| `--output` | Path to write tagged output (optional; prints to stdout if omitted) |
| `--output-dir` | Directory containing the trained adapter |

### `batch` â€” Process a directory of documents

```bash
python -m app.cli batch --input DIR [--output DIR] [--output-dir DIR]
```

| Option | Default | Description |
|---|---|---|
| `--input` | â€” | Directory containing `.txt` files (required) |
| `--output` | ./results | Output directory for tagged documents |
| `--output-dir` | ./checkpoints | Directory containing the adapter |

---

## 15. Limitations and Future Improvements

### Current Limitations

#### 1. Small training data
The built-in sample dataset has only 3â€“5 examples per class. With 20 classes and ~100 total examples, the model cannot truly learn to distinguish all clause types. LEDGAR mitigates this but is still small relative to the task complexity.

#### 2. Generative classification is suboptimal
Using a language model's next-token probabilities for classification is an indirect approach. A dedicated encoder model (e.g., Legal-BERT, RoBERTa-base) with a classification head would typically outperform this approach on accuracy.

#### 3. Single-token label representation
Each label must map to a single token. Labels like `<Limitation_of_Liability>` are added as special tokens and treated as atomic units, which works well but requires careful tokenizer management.

#### 4. No multi-label support
Real contracts sometimes overlap clause types (e.g., a clause that is both a Waiver and relates to Intellectual Property). The current model assigns exactly one label.

#### 5. CPU inference is slow
Without a GPU, inference for a 100-clause document may take 1â€“5 minutes. The model is not quantized.

#### 6. Hook-based LoRA merging
Because LoRA is applied via hooks rather than weight replacement, the LoRA delta cannot be "merged" into the base weights at save time (which would eliminate inference overhead). The current implementation always applies the delta at runtime.

### Future Improvements

| Improvement | Description | Priority |
|---|---|---|
| **Legal-BERT backbone** | Replace DistilGPT-2 with `nlpaueb/legal-bert-base-uncased` for stronger domain pre-training | High |
| **Weight merging** | Implement `merge_lora_into_weights()` to fold LoRA delta into base weights before export, eliminating hook overhead | Medium |
| **Quantization** | INT8/INT4 quantization (via `bitsandbytes`) for 4Ã— faster CPU inference | Medium |
| **Multi-label support** | Use sigmoid instead of softmax; predict multiple labels per clause | Medium |
| **Larger rank search** | Hyperparameter sweep over r âˆˆ {2, 4, 8, 16, 32} to find optimal rank-accuracy tradeoff | Low |
| **RLHF / DPO** | Use feedback from legal experts to further align predictions via reinforcement learning | Low |
| **Confidence calibration** | Platt scaling or temperature scaling to get well-calibrated confidence scores | Low |
| **REST API** | FastAPI wrapper for integration with document management systems | Medium |
| **Gradio UI** | Web interface for non-technical users to upload and tag contracts | Low |
| **Batch GPU inference** | Pipeline multiple clauses in a single forward pass for throughput | Medium |
| **Error analysis tool** | CLI command to show confusion matrix, most confused class pairs, example errors | Medium |

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest tests/ -v`)
4. Submit a pull request

### License

MIT License â€” see `LICENSE` for details.

---

*Built as a demonstration of parameter-efficient fine-tuning for legal NLP. Not intended for production legal advice.*

