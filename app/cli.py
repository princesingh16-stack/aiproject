"""Command-line interface for Legal Document LoRA Tagger."""

import argparse
import sys
import os


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_hello(args):
    print("Hello from Legal Document LoRA Tagger!")
    print("Version: 0.1.0")
    print("This tool classifies legal clauses using GPT-2 + LoRA.")


def cmd_prepare_data(args):
    """Prepare and inspect the dataset."""
    from app.data_prep import load_dataset_for_training

    use_ledgar = getattr(args, "ledgar", False)
    n_per_class = getattr(args, "n_per_class", 5)

    print(f"Preparing dataset (ledgar={use_ledgar}, n_per_class={n_per_class})...")
    train, val, test, source = load_dataset_for_training(
        use_ledgar=use_ledgar, n_per_class=n_per_class
    )
    print(f"Dataset source: {source}")
    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")
    print(f"\nSample train item:")
    if train:
        print(f"  Text: {train[0]['text'][:80]}...")
        print(f"  Label: {train[0]['label']}")


def cmd_train(args):
    """Train the LoRA model."""
    from app.model import load_tokenizer_and_model
    from app.lora import GPT2LoRAModel
    from app.data_prep import load_dataset_for_training
    from app.trainer import train_lora_model

    r = getattr(args, "lora_rank", 4)
    alpha = getattr(args, "lora_alpha", 16)
    epochs = getattr(args, "epochs", 3)
    batch_size = getattr(args, "batch_size", 4)
    lr = getattr(args, "lr", 3e-4)
    output_dir = getattr(args, "output_dir", "./checkpoints")
    n_per_class = getattr(args, "n_per_class", 5)
    use_ledgar = getattr(args, "ledgar", False)

    print("Loading model...")
    tokenizer, base_model = load_tokenizer_and_model()

    print(f"Applying LoRA (r={r}, alpha={alpha})...")
    model = GPT2LoRAModel(base_model, r=r, alpha=alpha)
    model.print_trainable_parameters()

    print("Loading dataset...")
    train_data, val_data, test_data, source = load_dataset_for_training(
        use_ledgar=use_ledgar, n_per_class=n_per_class
    )
    print(f"Dataset: {source} | Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    print(f"Training for {epochs} epochs...")
    results = train_lora_model(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    # Save final adapter
    model.save_adapter(os.path.join(output_dir, "final_adapter"))

    print("\n=== Training Complete ===")
    print(f"Training time: {results['training_time']:.1f}s")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    for i, (tl, vl) in enumerate(zip(results["train_losses"], results["val_losses"])):
        print(f"  Epoch {i+1}: train={tl:.4f}, val={vl:.4f}")


def cmd_evaluate(args):
    """Evaluate the trained model."""
    from app.model import load_tokenizer_and_model
    from app.lora import GPT2LoRAModel, load_lora_model
    from app.data_prep import load_dataset_for_training
    from app.evaluate import evaluate_model, evaluate_baseline, report_model_efficiency

    output_dir = getattr(args, "output_dir", "./checkpoints")
    adapter_path = os.path.join(output_dir, "final_adapter", "lora_adapter.pt")
    n_per_class = getattr(args, "n_per_class", 5)

    print("Loading model...")
    tokenizer, base_model = load_tokenizer_and_model()

    if os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}...")
        model = load_lora_model(base_model, adapter_path)
    else:
        print("No trained adapter found. Using base model (untrained).")
        model = GPT2LoRAModel(base_model)

    report_model_efficiency(model)

    print("Loading test data...")
    _, _, test_data, source = load_dataset_for_training(n_per_class=n_per_class)

    print("Evaluating baseline...")
    baseline = evaluate_baseline(test_data)
    print(f"Baseline (majority class '{baseline['majority_label']}'): {baseline['accuracy']:.4f} accuracy")

    print(f"Evaluating model on {len(test_data)} test samples...")
    metrics = evaluate_model(model, tokenizer, test_data)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Avg inference time: {metrics['avg_inference_time']*1000:.1f}ms per clause")


def cmd_predict(args):
    """Predict label for a single clause."""
    from app.inference import load_model_for_inference, predict_clause

    text = args.text
    output_dir = getattr(args, "output_dir", "./checkpoints")
    adapter_path = os.path.join(output_dir, "final_adapter", "lora_adapter.pt")

    print(f"Input: {text[:100]}")
    model, tokenizer = load_model_for_inference(output_dir, adapter_path=adapter_path)
    result = predict_clause(text, model, tokenizer)

    print(f"\nPREDICTED LABEL: {result['label']}")
    if result.get("confidence"):
        print(f"CONFIDENCE: {result['confidence']*100:.1f}%")


def cmd_process_document(args):
    """Process a full document and tag each clause."""
    from app.inference import (
        load_model_for_inference,
        predict_document,
        format_tagged_document,
        split_document_into_clauses,
    )

    input_file = args.input
    output_file = getattr(args, "output", None)
    output_dir = getattr(args, "output_dir", "./checkpoints")
    adapter_path = os.path.join(output_dir, "final_adapter", "lora_adapter.pt")

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    clauses = split_document_into_clauses(text)
    print(f"Found {len(clauses)} clauses in document.")

    model, tokenizer = load_model_for_inference(output_dir, adapter_path=adapter_path)
    results = predict_document(clauses, model, tokenizer)
    tagged = format_tagged_document(results)

    if output_file:
        from app.utils import ensure_dir
        ensure_dir(os.path.dirname(output_file) or ".")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(tagged)
        print(f"Tagged document saved to {output_file}")
    else:
        print(tagged)


def cmd_batch(args):
    """Batch process multiple documents."""
    from app.inference import (
        load_model_for_inference,
        predict_document,
        format_tagged_document,
        split_document_into_clauses,
    )
    from app.utils import ensure_dir

    input_dir = args.input
    output_dir_docs = getattr(args, "output", "./results")
    output_dir_model = getattr(args, "output_dir", "./checkpoints")
    adapter_path = os.path.join(output_dir_model, "final_adapter", "lora_adapter.pt")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    ensure_dir(output_dir_docs)
    model, tokenizer = load_model_for_inference(output_dir_model, adapter_path=adapter_path)

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    print(f"Processing {len(txt_files)} documents...")

    for fname in txt_files:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir_docs, fname.replace(".txt", "_tagged.txt"))
        with open(in_path, "r", encoding="utf-8") as f:
            text = f.read()
        clauses = split_document_into_clauses(text)
        results = predict_document(clauses, model, tokenizer)
        tagged = format_tagged_document(results)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(tagged)
        print(f"  {fname} -> {out_path}")

    print("Batch processing complete.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Legal Document LoRA Tagger CLI",
    )
    subparsers = parser.add_subparsers(dest="command")

    # hello
    subparsers.add_parser("hello", help="Print greeting")

    # prepare-data
    p = subparsers.add_parser("prepare-data", help="Prepare and inspect dataset")
    p.add_argument("--ledgar", action="store_true", help="Try to load LEDGAR dataset")
    p.add_argument("--n-per-class", type=int, default=5, dest="n_per_class")

    # train
    p = subparsers.add_parser("train", help="Train LoRA model")
    p.add_argument("--lora-rank", type=int, default=4, dest="lora_rank")
    p.add_argument("--lora-alpha", type=int, default=16, dest="lora_alpha")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", default="./checkpoints", dest="output_dir")
    p.add_argument("--n-per-class", type=int, default=5, dest="n_per_class")
    p.add_argument("--ledgar", action="store_true")

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate model")
    p.add_argument("--output-dir", default="./checkpoints", dest="output_dir")
    p.add_argument("--n-per-class", type=int, default=5, dest="n_per_class")

    # predict
    p = subparsers.add_parser("predict", help="Predict label for a clause")
    p.add_argument("--text", required=True, help="Clause text to classify")
    p.add_argument("--output-dir", default="./checkpoints", dest="output_dir")

    # process-document
    p = subparsers.add_parser("process-document", help="Tag all clauses in a document")
    p.add_argument("--input", required=True, help="Input document path")
    p.add_argument("--output", help="Output tagged document path")
    p.add_argument("--output-dir", default="./checkpoints", dest="output_dir")

    # batch
    p = subparsers.add_parser("batch", help="Batch tag documents in a directory")
    p.add_argument("--input", required=True, help="Input directory")
    p.add_argument("--output", default="./results", help="Output directory")
    p.add_argument("--output-dir", default="./checkpoints", dest="output_dir")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    command_map = {
        "hello": cmd_hello,
        "prepare-data": cmd_prepare_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
        "process-document": cmd_process_document,
        "batch": cmd_batch,
    }

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    fn = command_map.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
