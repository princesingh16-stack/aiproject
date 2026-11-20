"""Training loop for Legal LoRA Tagger."""

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW


def train_lora_model(
    model,
    tokenizer,
    train_data,
    val_data,
    output_dir="./checkpoints",
    num_epochs=3,
    batch_size=4,
    learning_rate=3e-4,
    max_length=128,
    device=None,
    log_every=10,
):
    """
    Train the LoRA model using a PyTorch training loop.
    Returns: dict with keys: train_losses, val_losses, training_time, best_val_loss.
    """
    from app.data_prep import LegalClauseDataset
    from app.utils import get_device, ensure_dir, get_logger

    logger = get_logger(__name__)
    if device is None:
        device = get_device()

    ensure_dir(output_dir)

    # Create datasets
    train_dataset = LegalClauseDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = LegalClauseDataset(val_data, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    logger.info(f"Starting training on {device}")
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(num_epochs):
        # ----------------------------------------------------------------
        # Training phase
        # ----------------------------------------------------------------
        model.train()
        epoch_train_loss = 0.0
        n_train_steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_steps += 1

            if step % log_every == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} Step {step} Train Loss: {loss.item():.4f}"
                )

        avg_train_loss = epoch_train_loss / max(n_train_steps, 1)
        train_losses.append(avg_train_loss)

        # ----------------------------------------------------------------
        # Validation phase
        # ----------------------------------------------------------------
        model.eval()
        epoch_val_loss = 0.0
        n_val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                epoch_val_loss += outputs.loss.item()
                n_val_steps += 1

        avg_val_loss = epoch_val_loss / max(n_val_steps, 1)
        val_losses.append(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Save checkpoint if best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(output_dir, "best_checkpoint")
            ensure_dir(ckpt_path)
            if hasattr(model, "save_adapter"):
                model.save_adapter(ckpt_path)
            logger.info(f"New best val loss: {best_val_loss:.4f} — checkpoint saved")

    training_time = time.time() - start_time
    logger.info(f"Training complete in {training_time:.1f}s")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "training_time": training_time,
        "best_val_loss": best_val_loss,
    }
