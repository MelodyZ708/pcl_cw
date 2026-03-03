#!/usr/bin/env python3
"""
Baseline: RoBERTa-base Fine-tuning for PCL Binary Classification
=================================================================
复现官方 baseline（RoBERTa-base），不加任何额外改进。
官方 baseline 指标: dev F1=0.48, test F1=0.49 (PCL positive class)

用法:
    python3 baseline_train.py [--epochs 3] [--batch_size 16] [--lr 2e-5]

输出:
    results/baseline_dev.txt   - dev 集预测（每行 0 或 1）
    results/baseline_test.txt  - test 集预测（每行 0 或 1）
    BestModel/baseline/        - 保存的模型权重（供 error analysis 使用）
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR   = "/vol/bitbucket/mz325/projects/pcl_cw"
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(PROJECT_DIR, "results")
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "BestModel", "baseline")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
DEV_CSV   = os.path.join(PROCESSED_DIR, "dev.csv")
TEST_CSV  = os.path.join(PROCESSED_DIR, "test.csv")

os.makedirs(RESULTS_DIR,   exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================================================
# 超参数默认值（与官方 baseline 保持一致）
# ============================================================
DEFAULTS = {
    "model_name":  "roberta-base",
    "max_length":  128,
    "batch_size":  16,
    "epochs":      3,
    "lr":          2e-5,
    "warmup_ratio": 0.1,
    "seed":        42,
}


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  使用 CPU（训练会较慢，建议使用 GPU）")
    return dev


# ============================================================
# Dataset
# ============================================================
class PCLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


# ============================================================
# 评估（返回 loss、macro-F1、PCL-F1、预测列表）
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, has_labels=True):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        if has_labels and "labels" in batch:
            total_loss += outputs.loss.item()
            all_labels.extend(batch["labels"].cpu().numpy())

    if has_labels and all_labels:
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        pcl_f1   = f1_score(all_labels, all_preds, pos_label=1, average="binary")
        avg_loss = total_loss / len(loader)
        return avg_loss, macro_f1, pcl_f1, all_preds, all_labels
    return None, None, None, all_preds, None


# ============================================================
# 主训练流程
# ============================================================
def main(args):
    set_seed(args.seed)
    device = get_device()

    print("\n" + "=" * 60)
    print(" Baseline: RoBERTa-base Fine-tuning")
    print("=" * 60)
    print(f"  model:      {args.model_name}")
    print(f"  max_length: {args.max_length}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs:     {args.epochs}")
    print(f"  lr:         {args.lr}")
    print(f"  seed:       {args.seed}")

    # ----------------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------------
    print("\n[1/5] 加载数据...")
    df_train = pd.read_csv(TRAIN_CSV, dtype={"par_id": str})
    df_dev   = pd.read_csv(DEV_CSV,   dtype={"par_id": str})
    df_test  = pd.read_csv(TEST_CSV,  dtype={"par_id": str})

    # 过滤空文本
    for df in [df_train, df_dev, df_test]:
        df.drop(df[df["text"].isna() | (df["text"].str.strip() == "")].index, inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_dev.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    train_texts  = df_train["text"].tolist()
    train_labels = df_train["label"].tolist()
    dev_texts    = df_dev["text"].tolist()
    dev_labels   = df_dev["label"].tolist()
    test_texts   = df_test["text"].tolist()

    print(f"  train: {len(train_texts)} | dev: {len(dev_texts)} | test: {len(test_texts)}")
    print(f"  train PCL rate: {sum(train_labels)/len(train_labels)*100:.1f}%")

    # ----------------------------------------------------------
    # 2. Tokenizer & Dataset
    # ----------------------------------------------------------
    print("\n[2/5] 初始化 Tokenizer 和 Dataset...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    train_dataset = PCLDataset(train_texts, train_labels, tokenizer, args.max_length)
    dev_dataset   = PCLDataset(dev_texts,   dev_labels,   tokenizer, args.max_length)
    test_dataset  = PCLDataset(test_texts,  None,         tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ----------------------------------------------------------
    # 3. 模型、优化器、调度器
    # ----------------------------------------------------------
    print("\n[3/5] 初始化模型...")
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    ).to(device)

    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  总训练步数: {total_steps} | warmup 步数: {warmup_steps}")

    # ----------------------------------------------------------
    # 4. 训练循环
    # ----------------------------------------------------------
    print("\n[4/5] 开始训练...")
    best_pcl_f1   = 0.0
    best_epoch    = 0
    best_dev_preds = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n  Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        dev_loss, macro_f1, pcl_f1, dev_preds, dev_true = evaluate(
            model, dev_loader, device, has_labels=True
        )
        print(f"    train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")
        print(f"    dev macro-F1={macro_f1:.4f} | dev PCL-F1={pcl_f1:.4f}")

        if pcl_f1 > best_pcl_f1:
            best_pcl_f1    = pcl_f1
            best_epoch     = epoch
            best_dev_preds = dev_preds
            # 保存最佳模型
            model.save_pretrained(MODEL_SAVE_DIR)
            tokenizer.save_pretrained(MODEL_SAVE_DIR)
            print(f"    *** 新最佳模型已保存 (PCL-F1={pcl_f1:.4f}) ***")

    print(f"\n  训练完成！最佳 epoch={best_epoch}, 最佳 dev PCL-F1={best_pcl_f1:.4f}")

    # ----------------------------------------------------------
    # 5. 加载最佳模型，生成最终预测
    # ----------------------------------------------------------
    print("\n[5/5] 加载最佳模型，生成预测文件...")
    best_model = RobertaForSequenceClassification.from_pretrained(MODEL_SAVE_DIR).to(device)

    # Dev 集完整评估
    _, macro_f1, pcl_f1, dev_preds, dev_true = evaluate(
        best_model, dev_loader, device, has_labels=True
    )
    print(f"\n  === Dev 集最终评估 ===")
    print(f"  macro-F1 : {macro_f1:.4f}")
    print(f"  PCL-F1   : {pcl_f1:.4f}  (baseline 目标: 0.48)")
    print(f"\n  分类报告:")
    print(classification_report(dev_true, dev_preds,
                                 target_names=["No-PCL", "PCL"], digits=4))
    print(f"  混淆矩阵:")
    cm = confusion_matrix(dev_true, dev_preds)
    print(f"              Pred No-PCL  Pred PCL")
    print(f"  True No-PCL   {cm[0][0]:>8}  {cm[0][1]:>8}")
    print(f"  True PCL      {cm[1][0]:>8}  {cm[1][1]:>8}")

    # 保存 dev.txt（格式：每行一个 0 或 1）
    dev_out_path = os.path.join(RESULTS_DIR, "baseline_dev.txt")
    with open(dev_out_path, "w") as f:
        for p in dev_preds:
            f.write(f"{p}\n")
    print(f"\n  已保存: results/baseline_dev.txt ({len(dev_preds)} 行)")

    # Test 集预测
    _, _, _, test_preds, _ = evaluate(
        best_model, test_loader, device, has_labels=False
    )
    test_out_path = os.path.join(RESULTS_DIR, "baseline_test.txt")
    with open(test_out_path, "w") as f:
        for p in test_preds:
            f.write(f"{p}\n")
    print(f"  已保存: results/baseline_test.txt ({len(test_preds)} 行)")

    # 同时保存 dev 集预测的详细 CSV（供 error analysis 使用）
    df_dev_result = df_dev.copy()
    df_dev_result["pred_baseline"] = dev_preds
    df_dev_result.to_csv(
        os.path.join(RESULTS_DIR, "baseline_dev_predictions.csv"),
        index=False
    )
    print(f"  已保存: results/baseline_dev_predictions.csv（含详细预测，供 error analysis）")

    print("\n" + "=" * 60)
    print(f" Baseline 训练完成！")
    print(f" dev PCL-F1 = {pcl_f1:.4f}  (官方 baseline: 0.48)")
    if pcl_f1 >= 0.48:
        print(f" [PASS] 达到或超过官方 baseline！")
    else:
        print(f" [NOTE] 未达到官方 baseline，可尝试增加 epochs 或调整 lr。")
    print("=" * 60)


# ============================================================
# 命令行参数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RoBERTa-base Baseline for PCL")
    parser.add_argument("--model_name",  type=str,   default=DEFAULTS["model_name"])
    parser.add_argument("--max_length",  type=int,   default=DEFAULTS["max_length"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--lr",          type=float, default=DEFAULTS["lr"])
    parser.add_argument("--warmup_ratio",type=float, default=DEFAULTS["warmup_ratio"])
    parser.add_argument("--seed",        type=int,   default=DEFAULTS["seed"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)