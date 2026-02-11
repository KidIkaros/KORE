"""Training diagnostics for Mamba-3 JEPA.

Provides three visualization tools:
  1. State EKG — per-layer SSM hidden state L2 norms (health monitor)
  2. ANGN Heatmap — gate activation bar chart (input importance)
  3. Dreaming — decoded text from current predicted embedding (latent reconstruction)

All functions are W&B-aware but degrade gracefully when W&B is unavailable.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from models.angn import AdaptiveNeuralGate
    from models.y_decoder import Mamba3Decoder

logger = logging.getLogger(__name__)

# Thresholds for state norm alerts
STATE_NORM_EXPLOSION = 100.0
STATE_NORM_COLLAPSE = 0.001


# ═══════════════════════════════════════════════════════════════════════════
# 1. State EKG — Mamba Hidden State Health Monitor
# ═══════════════════════════════════════════════════════════════════════════

def log_state_norms(
    state_norms: list[float],
    step: int,
    wandb_run=None,
) -> dict[str, float]:
    """Log per-layer SSM state norms and check for explosion/collapse.

    Args:
        state_norms: List of L2 norms, one per Mamba layer.
        step: Current training step.
        wandb_run: Optional wandb run object for logging.

    Returns:
        Dict of logged metrics.
    """
    metrics = {}
    for i, norm in enumerate(state_norms):
        metrics[f"state_norm/layer_{i}"] = norm

    if state_norms:
        metrics["state_norm/max"] = max(state_norms)
        metrics["state_norm/min"] = min(state_norms)
        metrics["state_norm/mean"] = sum(state_norms) / len(state_norms)

        # Alert on explosion or collapse
        max_norm = max(state_norms)
        min_norm = min(state_norms)
        if max_norm > STATE_NORM_EXPLOSION:
            layer_idx = state_norms.index(max_norm)
            logger.warning(
                f"[Step {step}] STATE EXPLOSION: layer {layer_idx} norm={max_norm:.1f} "
                f"(threshold={STATE_NORM_EXPLOSION})"
            )
        if min_norm < STATE_NORM_COLLAPSE:
            layer_idx = state_norms.index(min_norm)
            logger.warning(
                f"[Step {step}] STATE COLLAPSE: layer {layer_idx} norm={min_norm:.6f} "
                f"(threshold={STATE_NORM_COLLAPSE})"
            )

    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANGN Gate Heatmap — Input Importance
# ═══════════════════════════════════════════════════════════════════════════

def log_angn_heatmap(
    angn: "AdaptiveNeuralGate",
    step: int,
    wandb_run=None,
) -> list[float]:
    """Capture and log ANGN gate activations as a bar chart.

    Args:
        angn: AdaptiveNeuralGate module (must have get_last_gate_activations()).
        step: Current training step.
        wandb_run: Optional wandb run object.

    Returns:
        List of mean gate activations per layer.
    """
    activations = angn.get_last_gate_activations()
    if not activations:
        return []

    # Log raw values
    metrics = {}
    for i, act in enumerate(activations):
        metrics[f"angn_gate/layer_{i}"] = act
    metrics["angn_gate/mean"] = sum(activations) / len(activations)

    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

        # Generate bar chart image
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 3))
            layers = list(range(len(activations)))
            ax.bar(layers, activations, color="#4C9AFF", edgecolor="#2171B5")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Mean Gate Activation")
            ax.set_title(f"ANGN Gate Activations (step {step})")
            ax.set_ylim(0, 1)
            ax.set_xticks(layers)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            plt.close(fig)

            import wandb
            wandb_run.log({"angn_heatmap": wandb.Image(buf)}, step=step)
        except ImportError:
            pass  # matplotlib not available

    return activations


# ═══════════════════════════════════════════════════════════════════════════
# 3. Dreaming — Latent Reconstruction
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def log_dream_text(
    pred_embed: torch.Tensor,
    decoder: "Mamba3Decoder",
    step: int,
    max_tokens: int = 32,
    wandb_run=None,
) -> list[str]:
    """Decode predicted embedding to text and log it.

    Takes the current predicted embedding from the JEPA forward pass,
    runs it through the Y-Decoder's greedy generation, and logs the
    decoded tokens as text.

    Args:
        pred_embed: (batch, embed_dim) predicted embedding (detached).
        decoder: Mamba3Decoder with generate() method.
        step: Current training step.
        max_tokens: Max tokens to generate per sample.
        wandb_run: Optional wandb run object.

    Returns:
        List of decoded strings (one per batch element, max 4).
    """
    decoder.eval()

    # Only decode first few samples to save compute
    n_samples = min(pred_embed.shape[0], 4)
    embed_subset = pred_embed[:n_samples].detach()

    token_sequences = decoder.generate(
        embed_subset,
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy for deterministic dreams
    )

    # Convert token IDs to readable strings
    dream_texts = []
    for seq in token_sequences:
        # Best-effort: printable ASCII range, replace others with '?'
        chars = [chr(t) if 32 <= t < 127 else "?" for t in seq]
        dream_texts.append("".join(chars))

    if wandb_run is not None:
        try:
            import wandb
            html = "<br>".join(
                f"<b>Sample {i}:</b> <code>{text}</code>"
                for i, text in enumerate(dream_texts)
            )
            wandb_run.log({"dream_text": wandb.Html(html)}, step=step)
        except ImportError:
            pass

    for i, text in enumerate(dream_texts):
        logger.info(f"[Step {step}] Dream #{i}: {text[:80]}")

    return dream_texts


# ═══════════════════════════════════════════════════════════════════════════
# Combined diagnostic step
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics(
    state_norms: list[float] | None,
    angn: "AdaptiveNeuralGate | None",
    pred_embed: torch.Tensor | None,
    decoder: "Mamba3Decoder | None",
    step: int,
    wandb_run=None,
    dream_max_tokens: int = 32,
) -> dict:
    """Run all enabled diagnostics in one call.

    Args:
        state_norms: Per-layer SSM state norms (from backbone forward).
        angn: ANGN module (or None to skip).
        pred_embed: Predicted embedding tensor (or None to skip dreaming).
        decoder: Y-Decoder module (or None to skip dreaming).
        step: Current training step.
        wandb_run: Optional wandb run object.
        dream_max_tokens: Max tokens for dream generation.

    Returns:
        Dict with all collected diagnostic data.
    """
    result = {}

    if state_norms is not None:
        result["state_norms"] = log_state_norms(state_norms, step, wandb_run)

    if angn is not None:
        result["gate_activations"] = log_angn_heatmap(angn, step, wandb_run)

    if pred_embed is not None and decoder is not None:
        result["dream_texts"] = log_dream_text(
            pred_embed, decoder, step, dream_max_tokens, wandb_run,
        )

    return result
