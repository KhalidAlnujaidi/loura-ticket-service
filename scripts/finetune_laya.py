"""Fine-tune Laya on the labelled tickets (single device: MPS or CPU).

Recipe: Laya's own RLCD loop — noisy-logit policy gradient against the
strictly proper scoring reward (`proper_reward`) plus soft cross-entropy
guidance — adapted from the shipped 2xT4 Kaggle notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) down to one
device and to this project's two `choice` questions (category, priority).

The trainer imports the question definitions from `app.laya_llm.QUESTIONS`
(the exact dict the service sends at inference) and refuses to start if
Laya's own public normalization (`Agent._to_internal`) disagrees with how
this script renders them — a train/inference format drift would silently
void the fine-tune.

Usage:
  python -m scripts.finetune_laya --out checkpoints/loura-tickets-v1
  python -m scripts.finetune_laya --out /tmp/smoke --epochs 1 --limit-items 16
  python -m scripts.finetune_laya --out checkpoints/holdout \
      --exclude-ids t-1004,t-1006,t-1009
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_agreement import (  # noqa: E402
    git_rev,
    load_real_tickets,
    load_synthetic_tickets,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune Laya on the labelled tickets.")
    ap.add_argument("--base", default="convaiinnovations/laya", help="base checkpoint (id or dir)")
    ap.add_argument("--out", default="checkpoints/loura-tickets", help="output checkpoint dir")
    ap.add_argument("--tag", default="loura-tickets", help="model_name written into the config")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--exclude-ids", default="",
                    help="comma-separated real ticket ids to leave out of training (holdout arm)")
    ap.add_argument("--synthetic", choices=["train", "all", "none"], default="train",
                    help="which synthetic rows to train on (default: the train split)")
    ap.add_argument("--limit-items", type=int, default=0, help="cap items (smoke runs)")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4, help="GRPO baseline samples")
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1.0e-4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--calib-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for memory (off by default: these sequences are short)")
    return ap.parse_args()


def pick_device(name: str):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def internal_questions():
    """The inference-side internal question dicts, verified against Laya's own code."""
    from laya.agent import Agent

    from app.laya_llm import QUESTIONS

    internal = {qid: Agent._to_internal(qdef) for qid, qdef in QUESTIONS.items()}
    # Fail closed on drift: if the package normalizes differently, our training
    # sequences would not be the sequences the service actually sends.
    for qid, qdef in QUESTIONS.items():
        expected = {"t": qdef["type"], "ins": qdef["instructions"], "crit": qdef["criteria"]}
        if internal[qid] != expected:
            raise SystemExit(
                f"laya's normalization of question {qid!r} differs from this script's: "
                f"{internal[qid]!r} != {expected!r} — refusing to train on a drifted format")
    return internal


def build_items(tickets: list[dict], tok, cfg: dict, internal: dict) -> list[dict]:
    from laya.common import QTYPES, build_sequence, render_options

    items = []
    for t in tickets:
        state = {"subject": t["subject"], "body": t["body"]}  # exactly what LayaLLM sends
        for qid, q in internal.items():
            gold = t["category"] if qid == "category" else t["priority"]
            keys = list(q["crit"])
            if gold not in keys:
                raise SystemExit(f"{t['id']}: label {gold!r} is not an option of {qid!r} ({keys})")
            seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise SystemExit(f"{t['id']}: options exceed head_max_len for {qid!r}")
            target = [1.0 if k == gold else 0.0 for k in keys]
            items.append({
                "ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                "target": target, "label": keys.index(gold),
                "ticket": t["id"], "question": qid,
            })
    return items


def collate(items: list[dict], pad_id: int):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos,
        "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def fit_one_temp(sel: list) -> float:
    """Fit a single temperature by minimising NLL on (logits, target) pairs."""
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    t_start = time.time()
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ---- data -------------------------------------------------------------
    excluded = {i.strip() for i in args.exclude_ids.split(",") if i.strip()}
    real = [t for t in load_real_tickets(Path(args.data_dir)) if t["id"] not in excluded]
    synthetic = load_synthetic_tickets(Path(args.data_dir))
    if args.synthetic == "train":
        synthetic = [t for t in synthetic if t["split"] == "train"]
    elif args.synthetic == "none":
        synthetic = []
    tickets = real + synthetic
    if not tickets:
        raise SystemExit("no training tickets selected")

    # ---- model + tokenizer ------------------------------------------------
    print(f"[finetune] base={args.base} device={device} tickets={len(tickets)} "
          f"(real={len(real)}, synthetic={len(synthetic)}, excluded={sorted(excluded) or '-'})")
    # Only the base checkpoint's own files: the repo also carries `multilingual/`
    # and `typed-decisions/` variants (~800MB each) that this trainer never reads.
    model_dir = (snapshot_download(args.base,
                                   allow_patterns=["model.safetensors", "rl_agent_config.json",
                                                   "encoder/*", "tokenizer/*"])
                 if not Path(args.base).exists() else args.base)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    base_temperatures = list(cfg.get("temperature", [1.0, 1.0, 1.0]))

    internal = internal_questions()
    items = build_items(tickets, tok, cfg, internal)
    print(f"[finetune] built {len(items)} training items "
          f"({len(items) // len(internal)} tickets x {len(internal)} questions)")

    # ---- train/calib split (calib never enters a training batch) ----------
    order = list(range(len(items)))
    random.Random(args.seed).shuffle(order)
    n_calib = min(400, max(10, int(len(items) * args.calib_fraction)))
    calib_items = [items[i] for i in sorted(order[:n_calib])]
    train_items = [items[i] for i in sorted(order[n_calib:])]
    if args.limit_items:
        train_items = train_items[: args.limit_items]
    print(f"[finetune] {len(train_items)} train items, {len(calib_items)} held out for calibration")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    if args.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    enc_params = [p for n, p in model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=0.01)
    updates_per_epoch = max(1, (len(train_items) // (args.micro_batch * args.grad_accum)))
    total_updates = updates_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_updates), eta_min=1e-6)

    def evaluate(batch_items: list[dict]) -> tuple[float, list]:
        """Accuracy + raw logits on a list of items (no grad)."""
        model.eval()
        preds, correct = [], 0
        with torch.no_grad():
            for c in range(0, len(batch_items), 16):
                chunk = batch_items[c:c + 16]
                b = collate(chunk, tok.pad_token_id)
                logits, _ = model(
                    b["input_ids"].to(device), b["attention_mask"].to(device),
                    b["marker_pos"].to(device), b["marker_mask"].to(device),
                    b["qtype"].to(device))
                logits = logits.float().cpu().numpy()
                for r, it in enumerate(chunk):
                    k = len(it["markers"])
                    row = logits[r, :k]
                    preds.append((it["qtype"], row, it["target"]))
                    if int(np.argmax(row)) == it["label"]:
                        correct += 1
        model.train()
        return (correct / len(batch_items) if batch_items else 0.0), preds

    loss_curve: list[dict] = []
    for epoch in range(args.epochs):
        random.seed(args.seed + epoch)
        random.shuffle(train_items)
        progress = epoch / max(1, args.epochs - 1) if args.epochs > 1 else 0.0
        sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
        epoch_loss, n_batches, reward_sum = 0.0, 0, 0.0
        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for b_idx in range(0, len(train_items), args.micro_batch):
            chunk = train_items[b_idx:b_idx + args.micro_batch]
            if not chunk:
                continue
            b = collate(chunk, tok.pad_token_id)
            logits, act = model(
                b["input_ids"].to(device), b["attention_mask"].to(device),
                b["marker_pos"].to(device), b["marker_mask"].to(device),
                b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)

            # 1. sample G noisy logit vectors (zero-mean projection onto the mask)
            eps = torch.randn((args.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            # 2. strictly proper reward, normalised into an advantage
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask,
                                  w_sph=args.w_sph, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            # 3. policy gradient + soft cross-entropy guidance
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / args.grad_accum + 0.0 * act.sum()

            loss.backward()
            accum += 1
            if accum % args.grad_accum == 0 or (b_idx + args.micro_batch) >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.item()) * args.grad_accum
            reward_sum += float(r.mean().item())
            n_batches += 1

        calib_acc, _ = evaluate(calib_items)
        mean_loss = epoch_loss / max(1, n_batches)
        mean_reward = reward_sum / max(1, n_batches)
        loss_curve.append({"epoch": epoch + 1, "loss": round(mean_loss, 4),
                           "reward": round(mean_reward, 4), "calib_acc": round(calib_acc, 4),
                           "sigma": round(sigma, 3),
                           "seconds": round(time.time() - t_epoch, 1)})
        print(f"[finetune] epoch {epoch + 1}/{args.epochs} loss={mean_loss:.4f} "
              f"reward={mean_reward:.3f} calib_acc={calib_acc:.3f} "
              f"sigma={sigma:.2f} ({time.time() - t_epoch:.0f}s)")

    # ---- post-training temperature calibration on the held-out slice ------
    _, calib_preds = evaluate(calib_items)
    fitted = list(base_temperatures)
    fit_note = "kept base temperature (fewer than 10 calibration items for this type)"
    sel = [(z, t) for qtype, z, t in calib_preds if qtype == 0]
    if len(sel) >= 10:
        fitted[0] = round(fit_one_temp(sel), 4)
        fit_note = f"fitted on {len(sel)} held-out choice items"
    print(f"[finetune] temperatures {base_temperatures} -> {fitted} ({fit_note})")

    # ---- save checkpoint --------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    model.encoder.config.save_pretrained(str(out / "encoder"))
    tok.save_pretrained(str(out / "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = args.tag
    cfg["temperature"] = fitted
    # The fitted value is per question type; a stale per-bucket override would hide it.
    cfg.pop("temperature_by_options", None)
    cfg["fine_tune_provenance"] = {
        "base": args.base,
        "data": {
            "real": [t["id"] for t in real],
            "excluded_real": sorted(excluded),
            "synthetic": {"count": len(synthetic), "split": args.synthetic},
        },
        "hyperparameters": {k: getattr(args, k) for k in (
            "epochs", "micro_batch", "grad_accum", "group_size", "lr_encoder", "lr_head",
            "sigma_start", "sigma_end", "w_sph", "calib_fraction", "seed")},
        "items": {"train": len(train_items), "calib": len(calib_items)},
        "temperature_note": fit_note,
    }
    with open(out / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_rev": git_rev(),
        "base": args.base, "out": str(out), "device": str(device),
        "tickets": {"real": len(real), "synthetic": len(synthetic), "excluded_real": sorted(excluded)},
        "items": {"train": len(train_items), "calib": len(calib_items)},
        "loss_curve": loss_curve,
        "temperatures": {"base": base_temperatures, "fitted": fitted, "note": fit_note},
        "wall_seconds": round(time.time() - t_start, 1),
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"[finetune] checkpoint -> {out} ({report['wall_seconds']:.0f}s wall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
