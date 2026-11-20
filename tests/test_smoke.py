"""Smoke tests — fast end-to-end checks."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCLISmoke:
    def test_hello_command(self, capsys):
        from app.cli import cmd_hello
        cmd_hello(None)
        captured = capsys.readouterr()
        assert "Hello" in captured.out or "Legal" in captured.out

    def test_prepare_data_smoke(self, capsys):
        import argparse
        from app.cli import cmd_prepare_data
        args = argparse.Namespace(ledgar=False, n_per_class=3)
        cmd_prepare_data(args)
        captured = capsys.readouterr()
        assert "Train" in captured.out or "sample" in captured.out.lower()

    def test_cli_parser_builds(self):
        from app.cli import build_parser
        parser = build_parser()
        assert parser is not None


class TestImports:
    def test_app_imports(self):
        import app
        assert app.__version__ == "0.1.0"

    def test_data_prep_imports(self):
        from app.data_prep import CLAUSE_LABELS, SAMPLE_DATA
        assert CLAUSE_LABELS is not None
        assert SAMPLE_DATA is not None

    def test_utils_imports(self):
        from app.utils import (
            format_clause, clean_label, ensure_dir,
            get_logger, count_parameters,
        )
        assert callable(format_clause)
        assert callable(clean_label)

    def test_model_imports(self):
        from app.model import get_special_tokens, get_label_token_ids
        assert callable(get_special_tokens)

    def test_inference_utils_imports(self):
        from app.inference import (
            split_document_into_clauses, format_tagged_document
        )
        assert callable(split_document_into_clauses)

    def test_evaluate_imports(self):
        from app.evaluate import evaluate_baseline, report_model_efficiency
        assert callable(evaluate_baseline)


class TestDataIntegrity:
    def test_all_labels_have_samples(self):
        from app.data_prep import CLAUSE_LABELS, SAMPLE_DATA
        for label in CLAUSE_LABELS:
            assert label in SAMPLE_DATA
            assert len(SAMPLE_DATA[label]) >= 3

    def test_sample_texts_are_strings(self):
        from app.data_prep import SAMPLE_DATA
        for label, examples in SAMPLE_DATA.items():
            for ex in examples:
                assert isinstance(ex, str)
                assert len(ex) > 10, f"Sample too short for label {label}"

    def test_clause_labels_unique(self):
        from app.data_prep import CLAUSE_LABELS
        assert len(CLAUSE_LABELS) == len(set(CLAUSE_LABELS))

    def test_dataset_split_sizes(self):
        from app.data_prep import get_sample_dataset, CLAUSE_LABELS
        train, val, test = get_sample_dataset(n_per_class=5)
        total = len(train) + len(val) + len(test)
        expected_total = len(CLAUSE_LABELS) * 5
        assert total == expected_total


class TestBaselineEvaluator:
    def test_majority_class_baseline(self):
        from app.evaluate import evaluate_baseline
        test_data = [
            {"text": "text1", "label": "Governing_Law"},
            {"text": "text2", "label": "Governing_Law"},
            {"text": "text3", "label": "Termination"},
        ]
        result = evaluate_baseline(test_data)
        assert result["majority_label"] == "Governing_Law"
        assert abs(result["accuracy"] - 2 / 3) < 1e-6

    def test_baseline_empty_data(self):
        from app.evaluate import evaluate_baseline
        result = evaluate_baseline([])
        assert result["accuracy"] == 0
