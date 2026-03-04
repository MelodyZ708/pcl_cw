#!/usr/bin/env python3
"""
Best Model: Multi-Task Learning + Weighted CrossEntropyLoss for PCL Detection
==============================================================================
Two novelties on top of the RoBERTa-base baseline:

  Novelty 1 — Multi-Task Learning (MTL):
    Primary task:   Binary classification (PCL vs No-PCL)
    Auxiliary task: 5-class classification of orig_label (0-4, PCL intensity/confidence)
    Architecture:   Shared RoBERTa encoder with two independent classification heads
    Total loss:     (1 - aux_weight) * primary_loss + aux_weight * aux_loss

    Rationale:
      The orig_label field retains the annotators' graded judgement of PCL severity
      (0 = clearly no PCL, 4 = clearly PCL). The auxiliary task forces the shared
      encoder to distinguish borderline PCL (orig=2) from clear-cut PCL (orig=4),
      learning a more fine-grained semantic representation than the binary label alone.
      This is especially beneficial for classifying samples near the decision boundary.
      MTL-only result: dev PCL-F1 = 0.6124

  Novelty 2 — Weighted CrossEntropyLoss (primary task loss):
    CrossEntropyLoss(weight=[1.0, class_weight])
    Assigns a higher loss weight to the PCL class (minority class).

    Rationale:
      PCL accounts for only 9.5% of the training set (794/8372). This severe class
      imbalance causes the model to favour predicting No-PCL to minimise overall loss.
      Weighted CE directly amplifies the loss contribution of each misclassified PCL
      sample, forcing the model to attend more closely to the minority class and
      improving Recall. Compared to Focal Loss (dynamic weights), Weighted CE uses
      static weights that produce stable gradients compatible with MTL auxiliary gradients.

Usage:
    python3 best_model_train.py [--epochs 10] [--batch_size 16] [--lr 2e-5]
                                [--aux_weight 0.3] [--class_weight 3.0]
                                [--output_prefix bestmodel]

Outputs:
    results/bestmodel_dev.txt              - dev set predictions (0 or 1 per line, 2094 lines)
    results/bestmodel_test.txt             - test set predictions (0 or 1 per line)
    results/bestmodel_dev_predictions.csv  - detailed prediction CSV with probabilities
    BestModel/bestmodel/                   - saved model checkpoint

Note on dev.txt line count:
    The official dev split contains 2094 entries, but par_id 8640 has an empty text
    field and is removed during preprocessing, so the model produces 2093 predictions.
    This script automatically re-inserts a "0" prediction at the correct position so
    that dev.txt always has exactly 2094 lines, matching the official split expected
    by the evaluation script.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ============================================================
# Path configuration
# ============================================================
PROJECT_DIR   = "/vol/bitbucket/mz325/projects/pcl_cw"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(PROJECT_DIR, "results")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
DEV_CSV   = os.path.join(PROCESSED_DIR, "dev.csv")
TEST_CSV  = os.path.join(PROCESSED_DIR, "test.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# par_id of the empty-text entry removed during preprocessing
MISSING_PAR_ID  = 8640
DEV_TOTAL_LINES = 2094

# ============================================================
# Default hyperparameters
# ============================================================
DEFAULTS = {
    "model_name":    "roberta-base",
    "max_length":    128,
    "batch_size":    16,
    "epochs":        10,
    "lr":            2e-5,
    "warmup_ratio":  0.1,
    "aux_weight":    0.3,   # auxiliary task loss weight
    "class_weight":  3.0,   # PCL class weight (No-PCL=1.0)
    "seed":          42,
    "output_prefix": "bestmodel",
}

NUM_AUX_CLASSES = 5  # orig_label values: 0, 1, 2, 3, 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Using CPU (GPU recommended)")
    return dev


def fix_dev_predictions(preds, dev_csv_path):
    """
    The official dev split has DEV_TOTAL_LINES (2094) entries, but par_id=8640
    has an empty text field and was removed during preprocessing, so the model
    produces 2093 predictions. This function re-inserts a prediction of 0
    (No-PCL) at the correct position so that dev.txt has exactly 2094 lines.

    dev.csv is sorted by par_id ascending. We count how many rows have
    par_id < 8640; that is the 0-based insertion index.
    """
    if len(preds) == DEV_TOTAL_LINES:
        return list(preds)  

    df_dev     = pd.read_csv(dev_csv_path, dtype={"par_id": str})
    insert_pos = sum(1 for pid in df_dev["par_id"].tolist()
                     if pid.isdigit() and int(pid) < MISSING_PAR_ID)

    preds_fixed = list(preds)
    preds_fixed.insert(insert_pos, 0)

    print(f"  [dev.txt fix] Inserted '0' at line {insert_pos + 1} (par_id={MISSING_PAR_ID}, empty text).")
    print(f"  [dev.txt fix] Lines: {len(preds)} -> {len(preds_fixed)}")
    assert len(preds_fixed) == DEV_TOTAL_LINES, \
        f"Expected {DEV_TOTAL_LINES} lines after fix, got {len(preds_fixed)}"
    return preds_fixed


class PCLMultiTaskDataset(Dataset):
    def __init__(self, texts, binary_labels, aux_labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.binary_labels = (
            torch.tensor(binary_labels, dtype=torch.long)
            if binary_labels is not None else None
        )
        self.aux_labels = (
            torch.tensor(aux_labels, dtype=torch.long)
            if aux_labels is not None else None
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.binary_labels is not None:
            item["binary_labels"] = self.binary_labels[idx]
        if self.aux_labels is not None:
            item["aux_labels"] = self.aux_labels[idx]
        return item


# ============================================================
# Multi-Task model: shared RoBERTa encoder + two classification heads
# ============================================================
class RoBERTaMultiTask(nn.Module):
    def __init__(self, model_name: str, num_aux_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.roberta      = RobertaModel.from_pretrained(model_name)
        hidden_size       = self.roberta.config.hidden_size  # 768
        self.dropout      = nn.Dropout(dropout)
        self.primary_head = nn.Linear(hidden_size, 2)
        self.aux_head     = nn.Linear(hidden_size, num_aux_classes)

    def forward(self, input_ids, attention_mask):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        return self.primary_head(cls_output), self.aux_head(cls_output)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.roberta.save_pretrained(save_dir)
        torch.save(
            {"primary_head": self.primary_head.state_dict(),
             "aux_head":     self.aux_head.state_dict()},
            os.path.join(save_dir, "classification_heads.pt"),
        )

    @classmethod
    def from_pretrained(cls, save_dir: str, num_aux_classes: int = 5):
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        obj.roberta       = RobertaModel.from_pretrained(save_dir)
        hidden_size       = obj.roberta.config.hidden_size
        obj.dropout       = nn.Dropout(0.1)
        obj.primary_head  = nn.Linear(hidden_size, 2)
        obj.aux_head      = nn.Linear(hidden_size, num_aux_classes)
        heads = torch.load(
            os.path.join(save_dir, "classification_heads.pt"),
            map_location="cpu",
        )
        obj.primary_head.load_state_dict(heads["primary_head"])
        obj.aux_head.load_state_dict(heads["aux_head"])
        return obj


# ============================================================
# Train one epoch
# Primary task:   Weighted CrossEntropyLoss (higher weight for PCL)
# Auxiliary task: Standard CrossEntropyLoss
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device,
                primary_loss_fn, aux_weight: float):
    model.train()
    total_loss  = 0.0
    aux_loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        binary_labels  = batch.pop("binary_labels").to(device)
        aux_labels     = batch.pop("aux_labels").to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        primary_logits, aux_logits = model(input_ids, attention_mask)

        loss_primary = primary_loss_fn(primary_logits, binary_labels)
        loss_aux     = aux_loss_fn(aux_logits, aux_labels)
        loss = (1.0 - aux_weight) * loss_primary + aux_weight * loss_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Evaluation (primary task logits only; standard CE for dev loss comparison)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, has_labels=True):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss  = 0.0
    std_loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        binary_labels = batch.pop("binary_labels", None)
        batch.pop("aux_labels", None)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        primary_logits, _ = model(input_ids, attention_mask)
        probs = torch.softmax(primary_logits, dim=-1)[:, 1].cpu().numpy()
        preds = primary_logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(probs)

        if has_labels and binary_labels is not None:
            binary_labels = binary_labels.to(device)
            total_loss += std_loss_fn(primary_logits, binary_labels).item()
            all_labels.extend(binary_labels.cpu().numpy())

    if has_labels and all_labels:
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        pcl_f1   = f1_score(all_labels, all_preds, pos_label=1, average="binary")
        avg_loss = total_loss / len(loader)
        return avg_loss, macro_f1, pcl_f1, all_preds, all_labels, all_probs
    return None, None, None, all_preds, None, all_probs


# ============================================================
# Main training loop
# ============================================================
def main(args):
    set_seed(args.seed)
    device = get_device()

    model_save_dir = os.path.join(PROJECT_DIR, "BestModel", args.output_prefix)
    os.makedirs(model_save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f" Best Model: Multi-Task Learning + Weighted CE Loss [{args.output_prefix}]")
    print("=" * 70)
    print(f"  model:         {args.model_name}")
    print(f"  max_length:    {args.max_length}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  epochs:        {args.epochs}")
    print(f"  lr:            {args.lr}")
    print(f"  aux_weight:    {args.aux_weight}  (auxiliary task loss weight)")
    print(f"  class_weight:  {args.class_weight}  (PCL class weight, No-PCL=1.0)")
    print(f"  seed:          {args.seed}")
    print(f"  output_prefix: {args.output_prefix}")
    print(f"\n  Architecture: Shared RoBERTa encoder")
    print(f"        ├── Primary head: Linear(768, 2)  -> PCL vs No-PCL  [Weighted CE]")
    print(f"        └── Aux head:     Linear(768, 5)  -> orig_label 0-4 [CE]")
    print(f"  Total loss = {1-args.aux_weight:.1f} x WeightedCE(primary) + {args.aux_weight:.1f} x CE(aux)")
    print(f"\n  Novelty 1 (MTL):           aux_weight={args.aux_weight}")
    print(f"  Novelty 2 (Weighted CE):   weight=[No-PCL=1.0, PCL={args.class_weight}]")

    # ----------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------
    print("\n[1/5] Loading data...")
    df_train = pd.read_csv(TRAIN_CSV, dtype={"par_id": str})
    df_dev   = pd.read_csv(DEV_CSV,   dtype={"par_id": str})
    df_test  = pd.read_csv(TEST_CSV,  dtype={"par_id": str})

    for df in [df_train, df_dev, df_test]:
        df.drop(df[df["text"].isna() | (df["text"].str.strip() == "")].index, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_dev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    df_train["orig_label"] = pd.to_numeric(df_train["orig_label"], errors="coerce").fillna(0).astype(int).clip(0, 4)
    df_dev["orig_label"]   = pd.to_numeric(df_dev["orig_label"],   errors="coerce").fillna(0).astype(int).clip(0, 4)
    df_test["orig_label"]  = pd.to_numeric(df_test["orig_label"],  errors="coerce").fillna(0).astype(int).clip(0, 4)

    train_texts  = df_train["text"].tolist()
    train_binary = df_train["label"].tolist()
    train_aux    = df_train["orig_label"].tolist()
    dev_texts    = df_dev["text"].tolist()
    dev_binary   = df_dev["label"].tolist()
    dev_aux      = df_dev["orig_label"].tolist()
    test_texts   = df_test["text"].tolist()
    test_aux     = df_test["orig_label"].tolist()

    print(f"  train: {len(train_texts)} | dev: {len(dev_texts)} | test: {len(test_texts)}")
    print(f"  train PCL rate: {sum(train_binary)/len(train_binary)*100:.1f}%")
    print(f"  orig_label distribution (train): {pd.Series(train_aux).value_counts().sort_index().to_dict()}")

    # ----------------------------------------------------------
    # 2. Tokenizer & Dataset
    # ----------------------------------------------------------
    print("\n[2/5] Initialising tokenizer and datasets...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    train_dataset = PCLMultiTaskDataset(train_texts, train_binary, train_aux, tokenizer, args.max_length)
    dev_dataset   = PCLMultiTaskDataset(dev_texts,   dev_binary,   dev_aux,   tokenizer, args.max_length)
    test_dataset  = PCLMultiTaskDataset(test_texts,  None,         test_aux,  tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ----------------------------------------------------------
    # 3. Model, loss function, optimiser, scheduler
    # ----------------------------------------------------------
    print("\n[3/5] Initialising model...")
    model = RoBERTaMultiTask(args.model_name, num_aux_classes=NUM_AUX_CLASSES).to(device)

    # Primary task: Weighted CrossEntropyLoss with higher weight for PCL class
    class_weights = torch.tensor([1.0, args.class_weight], dtype=torch.float).to(device)
    primary_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"  Total training steps: {total_steps} | Warmup steps: {warmup_steps}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Primary task class weights: No-PCL=1.0, PCL={args.class_weight}")

    # ----------------------------------------------------------
    # 4. Training loop
    # ----------------------------------------------------------
    print("\n[4/5] Starting training...")
    best_pcl_f1    = 0.0
    best_epoch     = 0
    best_dev_preds = None
    best_dev_probs = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            primary_loss_fn, args.aux_weight,
        )
        dev_loss, macro_f1, pcl_f1, dev_preds, dev_true, dev_probs = evaluate(
            model, dev_loader, device, has_labels=True,
        )
        print(f"    train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")
        print(f"    dev macro-F1={macro_f1:.4f} | dev PCL-F1={pcl_f1:.4f}")

        if pcl_f1 > best_pcl_f1:
            best_pcl_f1    = pcl_f1
            best_epoch     = epoch
            best_dev_preds = dev_preds
            best_dev_probs = dev_probs
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            print(f"    *** New best model saved to BestModel/{args.output_prefix}/ (PCL-F1={pcl_f1:.4f}) ***")

    print(f"\n  Training complete. Best epoch={best_epoch}, best dev PCL-F1={best_pcl_f1:.4f}")

    # ----------------------------------------------------------
    # 5. Load best checkpoint and generate final predictions
    # ----------------------------------------------------------
    print("\n[5/5] Loading best checkpoint and generating prediction files...")
    best_model = RoBERTaMultiTask.from_pretrained(
        model_save_dir, num_aux_classes=NUM_AUX_CLASSES
    ).to(device)

    _, macro_f1, pcl_f1, dev_preds, dev_true, dev_probs = evaluate(
        best_model, dev_loader, device, has_labels=True,
    )

    print(f"\n  === Final Dev Set Evaluation ({args.output_prefix}) ===")
    print(f"  macro-F1 : {macro_f1:.4f}")
    print(f"  PCL-F1   : {pcl_f1:.4f}  (MTL-only: 0.6124 | our baseline: 0.6041 | official: 0.48)")
    print(f"\n  Classification report:")
    print(classification_report(dev_true, dev_preds, target_names=["No-PCL", "PCL"], digits=4))
    cm = confusion_matrix(dev_true, dev_preds)
    print(f"  Confusion matrix:")
    print(f"              Pred No-PCL  Pred PCL")
    print(f"  True No-PCL   {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"  True PCL      {cm[1][0]:>8}  {cm[1][1]:>8}")
    tp, fp, fn = cm[1][1], cm[0][1], cm[1][0]
    print(f"\n  TP={tp} | FP={fp} | FN={fn}")
    print(f"  Recall: {tp/(tp+fn+1e-9):.4f} | Precision: {tp/(tp+fp+1e-9):.4f}")

    # Re-insert prediction for par_id=8640 (empty text, removed during preprocessing)
    # so that dev.txt has exactly 2094 lines matching the official split.
    dev_preds_fixed = fix_dev_predictions(dev_preds, DEV_CSV)

    dev_out = os.path.join(RESULTS_DIR, f"{args.output_prefix}_dev.txt")
    with open(dev_out, "w") as f:
        for p in dev_preds_fixed:
            f.write(f"{p}\n")
    print(f"\n  Saved: results/{args.output_prefix}_dev.txt ({len(dev_preds_fixed)} lines)")

    _, _, _, test_preds, _, _ = evaluate(best_model, test_loader, device, has_labels=False)
    test_out = os.path.join(RESULTS_DIR, f"{args.output_prefix}_test.txt")
    with open(test_out, "w") as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"  Saved: results/{args.output_prefix}_test.txt ({len(test_preds)} lines)")

    # Save detailed prediction CSV (with probability scores for error analysis)
    df_dev_result = pd.read_csv(DEV_CSV, dtype={"par_id": str})
    df_dev_result = df_dev_result[
        df_dev_result["text"].notna() & (df_dev_result["text"].str.strip() != "")
    ].reset_index(drop=True)
    df_dev_result["pred"]     = dev_preds
    df_dev_result["prob_pcl"] = dev_probs
    csv_out = os.path.join(RESULTS_DIR, f"{args.output_prefix}_dev_predictions.csv")
    df_dev_result.to_csv(csv_out, index=False)
    print(f"  Saved: results/{args.output_prefix}_dev_predictions.csv")

    print("\n" + "=" * 70)
    print(f" Run complete: {args.output_prefix}")
    print(f" dev PCL-F1   = {pcl_f1:.4f}")
    print(f" dev macro-F1 = {macro_f1:.4f}")
    if pcl_f1 > 0.6124:
        print(f" [PASS] Outperforms MTL-only best (0.6124). Both novelties are effective!")
    elif pcl_f1 > 0.6041:
        print(f" [PASS] Outperforms our reproduced baseline (0.6041).")
    elif pcl_f1 >= 0.48:
        print(f" [PASS] Outperforms the official baseline (0.48).")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Best Model: Multi-Task Learning + Weighted CE Loss for PCL"
    )
    parser.add_argument("--model_name",    type=str,   default=DEFAULTS["model_name"])
    parser.add_argument("--max_length",    type=int,   default=DEFAULTS["max_length"])
    parser.add_argument("--batch_size",    type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",        type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    parser.add_argument("--warmup_ratio",  type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--aux_weight",    type=float, default=DEFAULTS["aux_weight"],
                        help="Auxiliary task loss weight. Total = (1-w)*WeightedCE + w*CE_aux.")
    parser.add_argument("--class_weight",  type=float, default=DEFAULTS["class_weight"],
                        help="PCL class weight in primary task loss. No-PCL=1.0. Recommended: 2.0~4.0")
    parser.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--output_prefix", type=str,   default=DEFAULTS["output_prefix"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)