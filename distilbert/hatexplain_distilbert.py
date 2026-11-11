"""DistilBERT multi-task classifier for HateXplain.

A single DistilBERT encoder is fine-tuned jointly with one linear head per
target (label + the 5 community columns). Same split, targets, metrics,
confusion matrices and sample inspection as nb/hatexplain_naive_bayes.py, so
both models can be compared side by side.
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import random
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from transformers.utils import logging as hf_logging

TRAIN_CSV = "../data/train.csv"
TEST_CSV = "../data/test.csv"
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = Path("model")
CM_DIR = Path("confusion_matrices")
TEXT_COL = "comment"
MAIN_TARGET = "label"
TARGET_COLS = ["label", "Race", "Religion", "Gender",
               "Sexual Orientation", "Miscellaneous"]
SEED = 42
N_SAMPLES = 15
DEFAULT_VALUES = {"No_race", "Nonreligious", "No_gender",
                  "No_orientation", "None"}
TOP_K_WORDS = 8
RETRAIN = False            # False = reuse the model saved in MODEL_DIR, if any

MAX_LEN = 128              # longest comment has 77 words (p99 = 52)
BATCH_SIZE = 32
EPOCHS = 4
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DROPOUT = 0.1
VAL_SIZE = 0.1             # slice of train used to pick the best epoch
CLASS_WEIGHT_POWER = 0.5   # 0 = no weighting, 1 = fully balanced weights
MAX_CLASS_WEIGHT = 10.0    # cap for very rare classes (e.g. Bisexual: 2 rows)
IG_STEPS = 32              # integrated-gradients interpolation steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = (torch.bfloat16 if USE_AMP and torch.cuda.is_bf16_supported()
             else torch.float16)

DROP_PLACEHOLDERS = r"<(user|number|url|email|phone)>"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(text: str) -> str:
    """Light cleaning: transformers rely on context and punctuation.

    Noise placeholders (<user>, <number>, ...) and URLs are removed; the
    meaningful ones (<happy>, <sad>, <censored>, ...) become plain words.
    """
    text = str(text).lower()
    text = re.sub(DROP_PLACEHOLDERS, " ", text)
    text = re.sub(r"<([a-z]+)>", r" \1 ", text)
    text = re.sub(r"http\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str) -> pd.DataFrame:
    """Load a split CSV, keep the original text, add a cleaned column."""
    df = pd.read_csv(path, keep_default_na=False)
    df[TARGET_COLS] = df[TARGET_COLS].replace("", "None")  # NaNs from older splits
    df["raw_text"] = df[TEXT_COL].astype(str)
    df[TEXT_COL] = df["raw_text"].map(clean_text)
    return df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)


class MultiTaskTransformer(nn.Module):
    """Shared encoder + one linear head per target, on the [CLS] vector."""

    def __init__(self, encoder: nn.Module, classes: dict):
        super().__init__()
        self.encoder = encoder
        self.classes = classes
        self.dropout = nn.Dropout(DROPOUT)
        hidden = encoder.config.hidden_size
        self.heads = nn.ModuleDict({self.key(t): nn.Linear(hidden, len(c))
                                    for t, c in classes.items()})

    @staticmethod
    def key(target: str) -> str:
        return target.replace(" ", "_")

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              inputs_embeds=inputs_embeds).last_hidden_state
        cls = self.dropout(hidden[:, 0])
        return {t: self.heads[self.key(t)](cls) for t in self.classes}


def encode_labels(df: pd.DataFrame, classes: dict) -> np.ndarray:
    """(n_rows, n_targets) class indices; unseen classes -> -100 (ignored)."""
    cols = [df[t].map({c: i for i, c in enumerate(classes[t])})
            .fillna(-100).astype(int).to_numpy() for t in TARGET_COLS]
    return np.stack(cols, axis=1)


def make_loader(texts, tokenizer, labels=None, shuffle=False) -> DataLoader:
    """Batches are tokenized on the fly with dynamic padding."""
    texts = list(texts)

    def collate(idx):
        batch = tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                          max_length=MAX_LEN, return_tensors="pt",
                          return_token_type_ids=False)
        if labels is not None:
            batch["labels"] = torch.from_numpy(labels[idx])
        return batch

    gen = torch.Generator().manual_seed(SEED)
    return DataLoader(range(len(texts)), batch_size=BATCH_SIZE, shuffle=shuffle,
                      collate_fn=collate, generator=gen)


def class_weights(df: pd.DataFrame, classes: dict) -> dict:
    """Dampened balanced weights: (n / (k * n_c)) ** power, capped."""
    weights = {}
    for t in TARGET_COLS:
        counts = df[t].value_counts().reindex(classes[t], fill_value=0)
        w = (len(df) / (len(counts) * np.maximum(counts.to_numpy(float), 1)))
        w = np.minimum(w ** CLASS_WEIGHT_POWER, MAX_CLASS_WEIGHT)
        weights[t] = torch.tensor(w, dtype=torch.float, device=DEVICE)
    return weights


@torch.no_grad()
def predict_proba(model: MultiTaskTransformer, tokenizer, texts) -> dict:
    """Class probabilities per target, as {target: (n_rows, n_classes)}."""
    model.eval()
    out = {t: [] for t in TARGET_COLS}
    for batch in make_loader(texts, tokenizer):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(**batch)
        for t in TARGET_COLS:
            out[t].append(torch.softmax(logits[t].float(), dim=-1).cpu().numpy())
    return {t: np.concatenate(p) for t, p in out.items()}


def predict(model: MultiTaskTransformer, tokenizer, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted class names per target, aligned with df.index."""
    probs = predict_proba(model, tokenizer, df[TEXT_COL])
    return pd.DataFrame({t: np.asarray(model.classes[t])[probs[t].argmax(1)]
                         for t in TARGET_COLS}, index=df.index)


def train_epoch(model, loader, losses, optimizer, scheduler, scaler) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            logits = model(**batch)
        loss = sum(losses[t](logits[t].float(), labels[:, i])
                   for i, t in enumerate(TARGET_COLS))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
    return total / len(loader)


def train_model(train_df: pd.DataFrame, tokenizer) -> MultiTaskTransformer:
    """Fine-tune on train only; a stratified slice of it picks the best epoch
    (mean macro-F1 over all targets). The test split is never touched here."""
    set_seed(SEED)
    classes = {t: [str(c) for c in sorted(train_df[t].unique())] for t in TARGET_COLS}
    fit_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED,
                                      stratify=train_df[MAIN_TARGET])
    print(f"[train] fit: {len(fit_df)} rows | val: {len(val_df)} rows | "
          f"device: {DEVICE} | amp: {AMP_DTYPE if USE_AMP else 'off'}")

    model = MultiTaskTransformer(AutoModel.from_pretrained(MODEL_NAME), classes).to(DEVICE)
    weights = class_weights(fit_df, classes)
    losses = {t: nn.CrossEntropyLoss(weight=weights[t]) for t in TARGET_COLS}
    loader = make_loader(fit_df[TEXT_COL], tokenizer,
                         encode_labels(fit_df, classes), shuffle=True)

    no_decay = lambda name: name.endswith("bias") or "norm" in name.lower()
    params = [
        {"params": [p for n, p in model.named_parameters() if not no_decay(n)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if no_decay(n)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=LR)
    total_steps = EPOCHS * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and AMP_DTYPE == torch.float16)

    best_score, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        loss = train_epoch(model, loader, losses, optimizer, scheduler, scaler)
        val_pred = predict(model, tokenizer, val_df)
        f1s = {t: f1_score(val_df[t], val_pred[t], average="macro", zero_division=0)
               for t in TARGET_COLS}
        score = float(np.mean(list(f1s.values())))
        print(f"[train] epoch {epoch}/{EPOCHS} | loss {loss:.4f} | "
              f"val macro-F1 {MAIN_TARGET} {f1s[MAIN_TARGET]:.4f} | "
              f"mean all targets {score:.4f} | {time.time() - start:.0f}s")
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    print(f"[train] best epoch = {best_epoch} (val mean macro-F1 = {best_score:.4f})")
    return model


def save_model(model: MultiTaskTransformer, tokenizer) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.encoder.config.save_pretrained(MODEL_DIR)
    torch.save({"classes": model.classes, "state_dict": model.state_dict()},
               MODEL_DIR / "model.pt")


def load_model():
    """Rebuild the model from MODEL_DIR (works offline)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ckpt = torch.load(MODEL_DIR / "model.pt", map_location="cpu")
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(MODEL_DIR))
    model = MultiTaskTransformer(encoder, ckpt["classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval(), tokenizer


def save_confusion_matrix(y_true, y_pred, target: str) -> None:
    """Print the confusion matrix and save it as PNG (counts + row-normalized).

    Rows = true class, columns = predicted class. The diagonal holds the
    correct predictions; everything off-diagonal is an error.
    """
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    CM_DIR.mkdir(exist_ok=True)
    size = max(5, 0.9 * len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 2, size + 1))
    for ax, norm, title in zip(axes, [None, "true"], ["Counts", "Normalized by true class"]):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, labels=labels, normalize=norm, ax=ax,
            cmap="Blues", xticks_rotation=45, colorbar=False,
            values_format="d" if norm is None else ".2f")
        ax.set_title(f"{target} - {title}")
    fig.tight_layout()
    fig.savefig(CM_DIR / f"cm_{target.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def evaluate_model(model: MultiTaskTransformer, tokenizer,
                   test_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every target on the held-out test split; return predictions."""
    preds = predict(model, tokenizer, test_df)
    for target in TARGET_COLS:
        print(f"\n===== {target} =====")
        print(f"accuracy: {accuracy_score(test_df[target], preds[target]):.4f} | "
              f"macro-F1: {f1_score(test_df[target], preds[target], average='macro'):.4f}")
        print(classification_report(test_df[target], preds[target],
                                    zero_division=0))
        save_confusion_matrix(test_df[target], preds[target], target)
    return preds


def explain(model: MultiTaskTransformer, tokenizer, text: str,
            k: int = TOP_K_WORDS, target: str = MAIN_TARGET):
    """Local XAI: predicted class + words that most favour it over runner-up.

    Integrated Gradients on the word embeddings: the logit margin
    (predicted - runner-up) is attributed to each token by integrating its
    gradient along a straight path from an all-zero embedding to the real one.
    Sub-word scores are summed per word, so "positive score = supports the
    predicted class", the same reading as the Naive Bayes explanation.
    """
    model.eval()
    enc = tokenizer(clean_text(text), truncation=True, max_length=MAX_LEN,
                    return_tensors="pt", return_token_type_ids=False)
    ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(input_ids=ids, attention_mask=mask)[target].float(), -1)[0]
    pred_idx, runner_idx = probs.argsort(descending=True)[:2].tolist()

    emb = model.encoder.get_input_embeddings()(ids).detach()          # (1, L, H)
    alphas = torch.linspace(1 / IG_STEPS, 1, IG_STEPS, device=DEVICE).view(-1, 1, 1)
    path = (alphas * emb).requires_grad_(True)                        # baseline = 0
    logits = model(inputs_embeds=path, attention_mask=mask.expand(IG_STEPS, -1))[target]
    margin = (logits[:, pred_idx] - logits[:, runner_idx]).sum()
    grads, = torch.autograd.grad(margin, path)
    scores = (emb[0] * grads.mean(0)).sum(-1).cpu().numpy()          # (L,)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    positions = {}
    for pos, word_id in enumerate(enc.word_ids(0)):
        if word_id is not None:
            positions.setdefault(word_id, []).append(pos)
    totals = {}
    for pos in positions.values():
        word = tokenizer.convert_tokens_to_string([tokens[p] for p in pos]).strip()
        if word not in tokenizer.all_special_tokens and any(ch.isalnum() for ch in word):
            totals[word] = totals.get(word, 0.0) + float(scores[pos].sum())
    words = sorted(totals.items(), key=lambda t: t[1], reverse=True)[:k]
    return model.classes[target][pred_idx], float(probs[pred_idx]), words


def inspect_samples(model, tokenizer, test_df, preds, n=N_SAMPLES):
    """Sample n test rows, compare prediction vs ground truth, explain them."""
    sample = test_df.sample(n=min(n, len(test_df)), random_state=SEED)
    n_ok = 0
    for idx, row in sample.iterrows():
        true, pred = row[MAIN_TARGET], preds.loc[idx, MAIN_TARGET]
        ok = true == pred
        n_ok += ok
        _, prob, words = explain(model, tokenizer, row["raw_text"])
        print(f"\n[{'OK ' if ok else 'ERR'}] {row['raw_text'][:200]}")
        print(f"   true={true} | pred={pred} ({prob:.2f})")
        shown = [f"{t}: {row[t]}->{preds.loc[idx, t]}" for t in TARGET_COLS[1:]
                 if not (row[t] in DEFAULT_VALUES
                         and preds.loc[idx, t] in DEFAULT_VALUES)]
        print("   communities (true->pred):", ", ".join(shown) or "none")
        print("   key words:", ", ".join(f"{w} ({s:.2f})" for w, s in words))
    print(f"\nSample accuracy ({MAIN_TARGET}): {n_ok}/{len(sample)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # emojis on Windows
    pd.set_option("display.width", 200)  # print wide confusion matrices in full
    hf_logging.set_verbosity_error()
    set_seed(SEED)
    train_df = load_data(TRAIN_CSV)
    test_df = load_data(TEST_CSV)
    print(f"train: {len(train_df)} rows | test: {len(test_df)} rows")
    print("Train class distribution:\n", train_df[MAIN_TARGET].value_counts(), "\n")

    if RETRAIN or not (MODEL_DIR / "model.pt").exists():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = train_model(train_df, tokenizer)
        save_model(model, tokenizer)
    else:
        model, tokenizer = load_model()
        print(f"[train] loaded saved model from {MODEL_DIR}/ (RETRAIN = True to retrain)")

    predictions = evaluate_model(model, tokenizer, test_df)
    inspect_samples(model, tokenizer, test_df, predictions)
