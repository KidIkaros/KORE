"""Phase 1: JEPA Pretraining — Predictor + Y-Encoder + ANGN.

Trains the Mamba-3 predictor to match Y-Encoder target embeddings via
InfoNCE contrastive loss. The X-Encoder (ViT) is frozen.

Usage:
    # Single GPU
    python scripts/train_jepa.py \
        --vit-checkpoint /path/to/vjepa2_vitl.pth \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1 \
        --epochs 50 --batch-size 64 --lr 3e-4

    # Multi-GPU (via accelerate)
    accelerate launch scripts/train_jepa.py \
        --vit-checkpoint /path/to/vjepa2_vitl.pth \
        --data-dir /path/to/image_text_pairs \
        --output-dir checkpoints/phase1

What trains:   Predictor backbone + ANGN gates + Y-Encoder
What's frozen: X-Encoder (V-JEPA 2 ViT-L)
What's off:    Y-Decoder, RecursionLayer
Loss:          InfoNCE (symmetric contrastive)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vljepa import Mamba3Jepa
from models.vit import VisionEncoder, VitConfig
from models.predictor import Mamba3Predictor
from models.angn import ANGNConfig
from models.y_encoder import Mamba3TextEncoder
from models.y_decoder import Mamba3Decoder
from utils.export_weights import export_to_safetensors
from utils.diagnostics import run_diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════

class ImageTextDataset(Dataset):
    """Placeholder dataset for image-text pairs.

    Replace with your actual dataset (CC3M, LAION, custom, etc.).
    Expected format: directory with pairs of files:
      {id}.jpg + {id}.txt  OR  a JSONL manifest.
    """

    def __init__(self, data_dir: str, image_size: int = 224, max_seq_len: int = 512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_seq_len = max_seq_len

        # Discover samples
        self.samples = []
        if (self.data_dir / "manifest.jsonl").exists():
            with open(self.data_dir / "manifest.jsonl") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            # Fall back to finding image files
            for ext in ["*.jpg", "*.png", "*.jpeg"]:
                for img_path in self.data_dir.glob(ext):
                    txt_path = img_path.with_suffix(".txt")
                    if txt_path.exists():
                        self.samples.append({
                            "image": str(img_path),
                            "text": str(txt_path),
                        })

        if not self.samples:
            print(f"[WARNING] No samples found in {data_dir}. Using synthetic data.")
            self.synthetic = True
        else:
            self.synthetic = False
            print(f"[Dataset] Found {len(self.samples)} image-text pairs in {data_dir}")

    def __len__(self) -> int:
        return max(len(self.samples), 1000) if self.synthetic else len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.synthetic:
            return self._synthetic_sample()

        sample = self.samples[idx]

        # Load image
        try:
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            img = Image.open(sample["image"]).convert("RGB")
            image_tensor = transform(img)
        except Exception:
            image_tensor = torch.randn(3, self.image_size, self.image_size)

        # Load text tokens (placeholder — replace with actual tokenizer)
        try:
            with open(sample["text"]) as f:
                text = f.read().strip()
            # Simple character-level tokenization (replace with real tokenizer)
            tokens = [ord(c) % 32000 for c in text[:self.max_seq_len]]
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
            token_tensor = torch.tensor(tokens, dtype=torch.long)
        except Exception:
            token_tensor = torch.randint(0, 32000, (self.max_seq_len,))

        return {
            "image": image_tensor,
            "tokens": token_tensor,
        }

    def _synthetic_sample(self) -> dict:
        """Generate a synthetic sample for testing the training loop."""
        return {
            "image": torch.randn(3, self.image_size, self.image_size),
            "tokens": torch.randint(0, 256, (self.max_seq_len,)),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epoch: int,
    temperature: float = 0.07,
    angn_reg_weight: float = 0.01,
    log_interval: int = 50,
    diag_interval: int = 200,
    wandb_run=None,
    autocast_ctx=None,
) -> dict:
    """Train one epoch of Phase 1 JEPA pretraining."""
    model.train()
    # Keep X-encoder frozen
    model.x_encoder.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_angn_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)

        # Query embeddings: use Y-encoder's embedding layer as query source
        # In practice, you'd have a separate query embedding — here we reuse
        # the token embeddings projected through query_proj
        with torch.no_grad():
            query_embeds = model.y_encoder.embedding(tokens)  # (B, L, d_model_enc)
            # Truncate/pad to match predictor's query_embed_dim
            qry_dim = model.predictor.query_proj.in_features
            if query_embeds.shape[-1] != qry_dim:
                # Project via simple truncation or zero-padding
                if query_embeds.shape[-1] > qry_dim:
                    query_embeds = query_embeds[..., :qry_dim]
                else:
                    pad = torch.zeros(*query_embeds.shape[:-1], qry_dim - query_embeds.shape[-1],
                                      device=device)
                    query_embeds = torch.cat([query_embeds, pad], dim=-1)

        # Determine if diagnostics are due this step
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        do_diag = diag_interval > 0 and (global_step + 1) % diag_interval == 0

        # Enable ANGN activation tracking only when diagnostics are due
        angn_module = getattr(model.predictor, "angn", None)
        if angn_module is not None and do_diag:
            angn_module.enable_activation_tracking(True)

        try:
            # Forward JEPA (with optional BF16 autocast)
            fwd_kwargs = dict(
                images=images,
                query_tokens=query_embeds,
                target_tokens=tokens,
                temperature=temperature,
                collect_state_norms=do_diag,
            )
            if autocast_ctx is not None:
                with autocast_ctx:
                    outputs = model.forward_jepa(**fwd_kwargs)
                    loss = outputs["loss"]
            else:
                outputs = model.forward_jepa(**fwd_kwargs)
                loss = outputs["loss"]

            # ANGN regularization (outside autocast — auxiliary term)
            angn_loss = torch.tensor(0.0, device=device)
            if model.predictor.angn is not None and angn_reg_weight > 0:
                angn_loss = model.predictor.angn.gate_regularization_loss()
                loss = loss + angn_reg_weight * angn_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += outputs["loss"].item()
            total_acc += outputs["accuracy"].item()
            total_angn_loss += angn_loss.item()
            n_batches += 1

            # Diagnostics (State EKG, ANGN heatmap, dreaming)
            if do_diag:
                run_diagnostics(
                    state_norms=outputs.get("state_norms"),
                    angn=angn_module,
                    pred_embed=outputs.get("pred_embed"),
                    decoder=model.y_decoder,
                    step=global_step,
                    wandb_run=wandb_run,
                )
        finally:
            # Always disable ANGN tracking to avoid leaked overhead on exception
            if angn_module is not None and do_diag:
                angn_module.enable_activation_tracking(False)

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * images.shape[0] / elapsed
            print(
                f"  [{epoch}][{batch_idx+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.3f} "
                f"angn_reg={total_angn_loss/n_batches:.4f} "
                f"({samples_per_sec:.1f} samples/s)"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_acc / max(n_batches, 1),
        "angn_reg": total_angn_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: Mamba3Jepa,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> dict:
    """Validate Phase 1 JEPA."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)

        query_embeds = model.y_encoder.embedding(tokens)
        qry_dim = model.predictor.query_proj.in_features
        if query_embeds.shape[-1] != qry_dim:
            if query_embeds.shape[-1] > qry_dim:
                query_embeds = query_embeds[..., :qry_dim]
            else:
                pad = torch.zeros(*query_embeds.shape[:-1], qry_dim - query_embeds.shape[-1],
                                  device=device)
                query_embeds = torch.cat([query_embeds, pad], dim=-1)

        outputs = model.forward_jepa(images, query_embeds, tokens, temperature)
        total_loss += outputs["loss"].item()
        total_acc += outputs["accuracy"].item()
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_accuracy": total_acc / max(n_batches, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1: JEPA Pretraining")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="Path to V-JEPA 2 ViT-L checkpoint (.pth or .safetensors)")
    parser.add_argument("--data-dir", type=str, default="data/image_text",
                        help="Directory containing image-text pairs")
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase1",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--angn-reg-weight", type=float, default=0.01,
                        help="ANGN gate sparsity regularization weight")
    parser.add_argument("--angn-enabled", action="store_true",
                        help="Enable ANGN gating during training")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny config for testing")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="mamba3-jepa")
    parser.add_argument("--diag-interval", type=int, default=200,
                        help="Run diagnostics every N steps (0 to disable)")
    parser.add_argument("--bf16", action="store_true",
                        help="Enable BF16 mixed precision (CUDA only)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (progressive seq length)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Device: {device}")

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    # BF16 mixed precision setup (no GradScaler — BF16 has same dynamic range as FP32)
    autocast_ctx = None
    if args.bf16:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            print("[Phase 1] BF16 mixed precision enabled")
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            print("[Phase 1] BF16 requested but not supported — falling back to FP32")

    # ─── Build model ───
    if args.tiny:
        print("[Phase 1] Using TINY config for testing")
        model = Mamba3Jepa.tiny()
        image_size = 16
        vocab_size = 256
    else:
        angn_config = ANGNConfig.small() if args.angn_enabled else ANGNConfig()
        if args.vit_checkpoint:
            x_encoder = VisionEncoder.from_vjepa2_checkpoint(args.vit_checkpoint)
        else:
            print("[Phase 1] No ViT checkpoint provided — using random ViT weights")
            x_encoder = VisionEncoder(VitConfig.vjepa2_vit_l())
            x_encoder.eval()
            for p in x_encoder.parameters():
                p.requires_grad = False

        model = Mamba3Jepa(
            x_encoder=x_encoder,
            predictor=Mamba3Predictor(
                d_model=1024, n_layers=12, d_state=128,
                expand=2, headdim=64, embed_dim=1536,
                vision_dim=1024, query_embed_dim=1024,
                angn_config=angn_config,
            ),
            y_encoder=Mamba3TextEncoder.small(),
            y_decoder=Mamba3Decoder.small(),
            shared_embed_dim=1536,
        )
        image_size = 224
        vocab_size = 32000

    model = model.to(device)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[Phase 1] Trainable: {trainable:,} | Frozen: {frozen:,}")

    # ─── Data ───
    dataset = ImageTextDataset(
        args.data_dir,
        image_size=image_size,
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ─── Optimizer ───
    # Only optimize trainable parameters (excludes frozen ViT)
    param_groups = [
        {"params": [p for p in model.predictor.parameters() if p.requires_grad], "lr": args.lr},
        {"params": [p for p in model.y_encoder.parameters() if p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / max(total_steps, 1),
    )

    # ─── Training ───
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    prev_seq_len = args.max_seq_len

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Curriculum learning: progressive sequence length
        # Recreate dataset + dataloader on phase change so forked workers see the new seq_len
        if args.curriculum:
            progress = epoch / args.epochs
            if progress < 1/3:
                curr_seq_len = max(128, args.max_seq_len // 4)
                phase_name = "short"
            elif progress < 2/3:
                curr_seq_len = max(256, args.max_seq_len // 2)
                phase_name = "medium"
            else:
                curr_seq_len = args.max_seq_len
                phase_name = "full"
            if curr_seq_len != prev_seq_len:
                dataset = ImageTextDataset(
                    args.data_dir, image_size=image_size, max_seq_len=curr_seq_len,
                )
                dataloader = DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True,
                )
                prev_seq_len = curr_seq_len
                print(f"  Curriculum: {phase_name} (seq_len={curr_seq_len})")

        train_metrics = train_one_epoch(
            model, dataloader, optimizer, scheduler, device,
            epoch, args.temperature, args.angn_reg_weight,
            diag_interval=args.diag_interval,
            wandb_run=wandb_run,
            autocast_ctx=autocast_ctx,
        )
        print(f"  Train: loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f}")

        # Checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = output_dir / f"phase1_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

        # Best model
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]
            best_path = output_dir / "phase1_best.pt"
            torch.save({"model_state_dict": model.state_dict()}, best_path)
            print(f"  New best model (loss={best_loss:.4f}) → {best_path}")

        # W&B
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, **train_metrics})

    # ─── Export to safetensors ───
    print(f"\n{'='*60}")
    print("Exporting weights to safetensors...")
    best_path = output_dir / "phase1_best.pt"
    if best_path.exists():
        export_to_safetensors(str(best_path), str(output_dir / "safetensors"), phase=1)

    print("\n[Phase 1] Training complete!")


if __name__ == "__main__":
    main()
