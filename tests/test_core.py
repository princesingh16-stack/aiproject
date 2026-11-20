"""Core unit tests for Legal Document LoRA Tagger."""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================
# Tests that do NOT require torch/transformers (pure Python)
# =========================================================

class TestDataPrep:
    def test_clause_labels_exist(self):
        from app.data_prep import CLAUSE_LABELS
        assert len(CLAUSE_LABELS) >= 10
        assert "Governing_Law" in CLAUSE_LABELS
        assert "Termination" in CLAUSE_LABELS

    def test_sample_data_coverage(self):
        from app.data_prep import SAMPLE_DATA, CLAUSE_LABELS
        for label in CLAUSE_LABELS:
            assert label in SAMPLE_DATA, f"Label {label} missing from SAMPLE_DATA"
            assert len(SAMPLE_DATA[label]) >= 2, f"Label {label} needs at least 2 samples"

    def test_get_sample_dataset_splits(self):
        from app.data_prep import get_sample_dataset
        train, val, test = get_sample_dataset(n_per_class=5, seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # No data leakage
        train_texts = {item["text"] for item in train}
        test_texts = {item["text"] for item in test}
        assert train_texts.isdisjoint(test_texts), "Data leakage: same text in train and test"

    def test_dataset_items_have_required_keys(self):
        from app.data_prep import get_sample_dataset, CLAUSE_LABELS
        train, val, test = get_sample_dataset(n_per_class=3)
        for item in train + val + test:
            assert "text" in item
            assert "label" in item
            assert item["label"] in CLAUSE_LABELS

    def test_load_dataset_for_training_returns_four_values(self):
        from app.data_prep import load_dataset_for_training
        result = load_dataset_for_training(use_ledgar=False, n_per_class=3)
        assert len(result) == 4
        train, val, test, source = result
        assert source in ("ledgar", "sample")

    def test_no_duplicates_across_splits(self):
        from app.data_prep import get_sample_dataset
        train, val, test = get_sample_dataset(n_per_class=5)
        train_set = {item["text"] for item in train}
        test_set = {item["text"] for item in test}
        val_set = {item["text"] for item in val}
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)


class TestUtils:
    def test_clean_label(self):
        from app.utils import clean_label
        assert clean_label("<Governing_Law>") == "Governing_Law"
        assert clean_label("<Payment Terms>") == "Payment_Terms"
        assert clean_label("Termination") == "Termination"

    def test_format_clause(self):
        from app.utils import format_clause
        result = format_clause("The agreement is binding.", "Governing_Law")
        assert "The agreement is binding." in result
        assert "Governing_Law" in result

    def test_ensure_dir(self, tmp_path):
        from app.utils import ensure_dir
        new_dir = str(tmp_path / "new_subdir")
        ensure_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_get_logger(self):
        from app.utils import get_logger
        logger = get_logger("test")
        assert logger is not None


class TestSpecialTokens:
    def test_special_tokens_format(self):
        from app.model import get_special_tokens
        from app.data_prep import CLAUSE_LABELS
        tokens = get_special_tokens()
        assert len(tokens) == len(CLAUSE_LABELS)
        for token in tokens:
            assert token.startswith("<")
            assert token.endswith(">")

    def test_special_tokens_no_duplicates(self):
        from app.model import get_special_tokens
        tokens = get_special_tokens()
        assert len(tokens) == len(set(tokens))


class TestInferenceUtils:
    def test_split_document(self):
        from app.inference import split_document_into_clauses
        doc = "Clause one text.\n\nClause two text.\n\nClause three."
        clauses = split_document_into_clauses(doc)
        assert len(clauses) == 3

    def test_format_tagged_document(self):
        from app.inference import format_tagged_document
        results = [
            {"text": "The agreement shall be governed by Indian law.", "label": "Governing_Law", "confidence": 0.75},
            {"text": "Either party may terminate this agreement.", "label": "Termination", "confidence": 0.60},
        ]
        output = format_tagged_document(results)
        assert "CLAUSE 1:" in output
        assert "Governing_Law" in output
        assert "CLAUSE 2:" in output
        assert "Termination" in output
        assert "CONFIDENCE:" in output


# =========================================================
# Tests that require torch (skipped gracefully if unavailable)
# =========================================================

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SKIP_TORCH = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


@SKIP_TORCH
class TestLoRALayer:
    def test_lora_layer_output_shape(self):
        import torch
        from app.lora import LoRALayer
        in_f, out_f, r = 64, 128, 4
        layer = LoRALayer(in_f, out_f, r=r, alpha=16)
        x = torch.randn(2, 10, in_f)
        out = layer(x)
        assert out.shape == (2, 10, out_f)

    def test_lora_b_initialized_to_zero(self):
        import torch
        from app.lora import LoRALayer
        layer = LoRALayer(64, 128, r=4, alpha=16)
        assert torch.all(layer.lora_B == 0)

    def test_lora_a_nonzero(self):
        import torch
        from app.lora import LoRALayer
        layer = LoRALayer(64, 128, r=4, alpha=16)
        assert not torch.all(layer.lora_A == 0)

    def test_lora_layer_trainable(self):
        from app.lora import LoRALayer
        layer = LoRALayer(64, 128, r=4, alpha=16)
        assert layer.lora_A.requires_grad
        assert layer.lora_B.requires_grad


@SKIP_TORCH
class TestGPT2LoRA:
    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        """Load a tiny model for testing."""
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained("distilgpt2")
            return tokenizer, model
        except Exception:
            pytest.skip("Could not download distilgpt2")

    def test_base_params_frozen(self, tiny_model_and_tokenizer):
        from app.lora import GPT2LoRAModel
        _, base_model = tiny_model_and_tokenizer
        lora_model = GPT2LoRAModel(base_model, r=4, alpha=16)
        for name, param in lora_model.base_model.named_parameters():
            assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_lora_params_trainable(self, tiny_model_and_tokenizer):
        from app.lora import GPT2LoRAModel
        _, base_model = tiny_model_and_tokenizer
        lora_model = GPT2LoRAModel(base_model, r=4, alpha=16)
        trainable = [p for p in lora_model.lora_layers.parameters() if p.requires_grad]
        assert len(trainable) > 0, "LoRA layers should have trainable parameters"

    def test_forward_pass(self, tiny_model_and_tokenizer):
        import torch
        from app.lora import GPT2LoRAModel
        tokenizer, base_model = tiny_model_and_tokenizer
        lora_model = GPT2LoRAModel(base_model, r=4, alpha=16)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        outputs = lora_model(input_ids=input_ids)
        assert outputs.logits is not None
        assert outputs.logits.shape[0] == 1

    def test_trainable_param_percentage(self, tiny_model_and_tokenizer):
        from app.lora import GPT2LoRAModel
        from app.utils import count_parameters
        _, base_model = tiny_model_and_tokenizer
        lora_model = GPT2LoRAModel(base_model, r=4, alpha=16)
        stats = count_parameters(lora_model)
        assert stats["trainable_pct"] < 10.0
        assert stats["trainable"] > 0
