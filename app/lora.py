"""Custom LoRA implementation for GPT-2 (no peft dependency)."""

import math
import os

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRALayer — the low-rank delta W' = W + (B @ A) * (alpha / r)
# ---------------------------------------------------------------------------

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer.
    Implements: output_delta = (dropout(x) @ A.T @ B.T) * scaling
    where scaling = alpha / r.

    A: shape (r, in_features)  — initialized kaiming_uniform
    B: shape (out_features, r) — initialized to zero (no perturbation at start)
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # result: (..., out_features)
        return (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


# ---------------------------------------------------------------------------
# LoRALinear — wraps nn.Linear; original weights frozen
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with LoRA adaptation."""

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False
        in_features = linear.in_features
        out_features = linear.out_features
        self.lora = LoRALayer(in_features, out_features, r=r, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights back into the base linear layer and return it."""
        with torch.no_grad():
            delta_W = (self.lora.lora_B @ self.lora.lora_A) * self.lora.scaling
            self.linear.weight.data += delta_W
        return self.linear


# ---------------------------------------------------------------------------
# GPT2LoRAModel — hook-based LoRA injection for distilgpt2 / gpt2
# ---------------------------------------------------------------------------

class GPT2LoRAModel(nn.Module):
    """
    GPT-2 model with LoRA adapters injected via forward hooks.

    GPT-2 uses Conv1D (not nn.Linear) internally. Using hooks avoids
    having to replace layers, which makes this approach robust to both
    GPT-2 Conv1D and standard nn.Linear.

    For each target module name (e.g. "c_attn", "c_proj"), a LoRALayer
    is created and a forward hook adds its delta to the module's output.
    """

    def __init__(
        self,
        base_model,
        r: int = 4,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules=None,
    ):
        super().__init__()
        self.base_model = base_model
        self.r = r
        self.alpha = alpha
        self.lora_layers = nn.ModuleDict()
        self._hooks = []

        if target_modules is None:
            target_modules = ["c_attn", "c_proj"]

        # Step 1: Freeze ALL base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Step 2: Inject LoRA layers and register hooks
        self._inject_lora_hooks(target_modules, r, alpha, dropout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_module_dimensions(self, module):
        """
        Return (in_features, out_features) for a GPT-2 Conv1D or nn.Linear.
        GPT-2 Conv1D stores weight as (in_features, out_features) and has .nf.
        nn.Linear stores weight as (out_features, in_features).
        """
        if hasattr(module, "nf"):  # GPT-2 Conv1D
            w = module.weight
            return w.shape[0], module.nf
        if hasattr(module, "weight"):
            w = module.weight
            if w.dim() == 2:
                return w.shape[1], w.shape[0]
        return None, None

    def _inject_lora_hooks(self, target_modules, r, alpha, dropout):
        for name, module in self.base_model.named_modules():
            leaf_name = name.split(".")[-1] if "." in name else name
            if leaf_name not in target_modules:
                continue
            in_f, out_f = self._get_module_dimensions(module)
            if in_f is None:
                continue

            safe_name = name.replace(".", "_")
            lora = LoRALayer(in_f, out_f, r=r, alpha=alpha, dropout=dropout)
            self.lora_layers[safe_name] = lora

            def make_hook(lora_layer):
                def hook(mod, input, output):
                    x = input[0]
                    delta = lora_layer(x)
                    if isinstance(output, tuple):
                        return (output[0] + delta,) + output[1:]
                    return output + delta
                return hook

            h = module.register_forward_hook(make_hook(lora))
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def remove_hooks(self):
        """Remove all forward hooks (e.g., before serialization)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, save_dir: str):
        """Save LoRA adapter weights and config."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "lora_state_dict": self.lora_layers.state_dict(),
                "r": self.r,
                "alpha": self.alpha,
                "config": {"r": self.r, "alpha": self.alpha},
            },
            os.path.join(save_dir, "lora_adapter.pt"),
        )
        print(f"LoRA adapter saved to {save_dir}/lora_adapter.pt")

    def merge_and_save(self, save_dir: str):
        """Save adapter weights (hook-based: separate from base model)."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            self.lora_layers.state_dict(),
            os.path.join(save_dir, "lora_adapters.pt"),
        )

    def print_trainable_parameters(self):
        from app.utils import count_parameters
        stats = count_parameters(self)
        print(f"Trainable parameters: {stats['trainable']:,}")
        print(f"Total parameters:     {stats['total']:,}")
        print(f"Trainable percentage: {stats['trainable_pct']:.4f}%")
        return stats


# ---------------------------------------------------------------------------
# load_lora_model helper
# ---------------------------------------------------------------------------

def load_lora_model(base_model, adapter_path: str, r: int = 4, alpha: int = 16, target_modules=None):
    """
    Create a GPT2LoRAModel, load saved adapter weights, and return the model.
    """
    model = GPT2LoRAModel(base_model, r=r, alpha=alpha, target_modules=target_modules)
    checkpoint = torch.load(adapter_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "lora_state_dict" in checkpoint:
        state_dict = checkpoint["lora_state_dict"]
        r_saved = checkpoint.get("r", r)
        alpha_saved = checkpoint.get("alpha", alpha)
    else:
        state_dict = checkpoint

    model.lora_layers.load_state_dict(state_dict, strict=False)
    return model
