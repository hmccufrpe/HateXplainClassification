import json
import re
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
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          get_linear_schedule_with_warmup)


TRAIN_CSV = "../data/train.csv"          
TEST_CSV = "../data/test.csv"
BERT_DIR = Path("bert_model")  
CM_DIR = Path("confusion_matrices_bert")
MODEL_NAME = "bert-base-uncased"
TEXT_COL = "comment"
MAIN_TARGET = "label"
TARGET_COLS = ["label", "Race", "Religion", "Gender",
               "Sexual Orientation", "Miscellaneous"]
DEFAULT_VALUES = {"No_race", "Nonreligious", "No_gender",
                  "No_orientation", "None"}  
MAX_LEN = 128                
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
VAL_SIZE = 0.1          
LOG_EVERY = 100          
SEED = 42
N_SAMPLES = 15       
TOP_K_WORDS = 8        
MAX_EXPLAIN_WORDS = 80   
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_text(text: str) -> str:
    """Light cleaning: keep the text natural for BERT, drop only noise."""
    text = str(text).lower()
    text = text.replace("<number>", "number").replace("<user>", "user")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: str) -> pd.DataFrame:
    """Load a split CSV, keep the original text, add a cleaned column."""
    # keep_default_na=False: the literal "None" (Miscellaneous) must not become NaN
    df = pd.read_csv(path, keep_default_na=False)
    df[TARGET_COLS] = df[TARGET_COLS].replace("", "None")
    df["raw_text"] = df[TEXT_COL].astype(str)
    df[TEXT_COL] = df["raw_text"].map(clean_text)
    return df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)


class HateDataset(Dataset):
    """Tokenized comments + one integer label per target."""

    def __init__(self, df: pd.DataFrame, tokenizer, class_maps: dict):
        self.enc = tokenizer(df[TEXT_COL].tolist(), truncation=True,
                             max_length=MAX_LEN)
        self.targets = np.stack(
            [df[t].map(class_maps[t]).to_numpy() for t in TARGET_COLS], axis=1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.targets[i].tolist()
        return item


def make_collate(tokenizer):
    """Dynamic padding: pad each batch to its longest sequence."""
    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return collate


class MultiHeadBert(nn.Module):
    """Shared BERT encoder + one linear head per target (on the [CLS] vector)."""

    def __init__(self, encoder, n_classes: dict):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList(
            [nn.Linear(encoder.config.hidden_size, n_classes[t])
             for t in TARGET_COLS])

    def forward(self, **inputs):
        cls = self.encoder(**inputs).last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return {t: head(cls) for t, head in zip(TARGET_COLS, self.heads)}


def class_weights(df: pd.DataFrame, target: str, class_map: dict):
    """Softened inverse-frequency weights (sqrt) to counter class imbalance."""
    counts = df[target].value_counts()
    w = np.ones(len(class_map))
    for cls, idx in class_map.items():
        w[idx] = (len(df) / (len(class_map) * max(counts.get(cls, 1), 1))) ** 0.5
    return torch.tensor(w, dtype=torch.float, device=DEVICE)


@torch.no_grad()
def predict_probs(model, tokenizer, texts: list, batch_size: int = 64) -> dict:
    """Softmax probabilities per target for already-cleaned texts."""
    model.eval()
    out = {t: [] for t in TARGET_COLS}
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True,
                        max_length=MAX_LEN, padding=True, return_tensors="pt")
        logits = model(**{k: v.to(DEVICE) for k, v in enc.items()})
        for t in TARGET_COLS:
            out[t].append(torch.softmax(logits[t], dim=-1).cpu().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def train_model(model, tokenizer, train_df, val_df, class_maps) -> None:
    """Multi-task fine-tuning; keeps the epoch with the best validation macro-F1."""
    loader = DataLoader(HateDataset(train_df, tokenizer, class_maps),
                        batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=make_collate(tokenizer))
    loss_fns = {t: nn.CrossEntropyLoss(weight=class_weights(train_df, t, class_maps[t]))
                for t in TARGET_COLS}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    val_true = val_df[MAIN_TARGET].map(class_maps[MAIN_TARGET]).to_numpy()
    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch)
            loss = sum(loss_fns[t](logits[t], labels[:, i])
                       for i, t in enumerate(TARGET_COLS))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % LOG_EVERY == 0:
                print(f"[train] epoch {epoch} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f}")

        val_pred = predict_probs(model, tokenizer, val_df[TEXT_COL].tolist())[MAIN_TARGET].argmax(1)
        val_f1 = f1_score(val_true, val_pred, average="macro")
        print(f"[val] epoch {epoch}: {MAIN_TARGET} macro-F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"[train] best validation macro-F1 = {best_f1:.4f}")


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


def evaluate_model(model, tokenizer, test_df, classes) -> pd.DataFrame:
    """Evaluate every head on the test split; return predicted class names."""
    probs = predict_probs(model, tokenizer, test_df[TEXT_COL].tolist())
    preds = pd.DataFrame(index=test_df.index)
    for target in TARGET_COLS:
        preds[target] = np.array(classes[target])[probs[target].argmax(1)]
        print(f"\n===== {target} =====")
        print(f"accuracy: {accuracy_score(test_df[target], preds[target]):.4f} | "
              f"macro-F1: {f1_score(test_df[target], preds[target], average='macro'):.4f}")
        print(classification_report(test_df[target], preds[target], zero_division=0))
        save_confusion_matrix(test_df[target], preds[target], target)
    preds["confidence"] = probs[MAIN_TARGET].max(axis=1)
    return preds


def explain(model, tokenizer, text: str, target: str = MAIN_TARGET,
            k: int = TOP_K_WORDS):
    """Local XAI by occlusion: probability drop when each word is removed.

    Returns (predicted class index, class probabilities, top-k (word, score)).
    """
    words = clean_text(text).split()
    n = min(len(words), MAX_EXPLAIN_WORDS)
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1:]) for i in range(n)]
    probs = predict_probs(model, tokenizer, variants)[target]
    pred_idx = int(probs[0].argmax())
    scores = probs[0, pred_idx] - probs[1:, pred_idx]
    ranked = sorted(zip(words[:n], scores), key=lambda t: -t[1])[:k]
    return pred_idx, probs[0], [(w, float(s)) for w, s in ranked]


def inspect_samples(model, tokenizer, test_df, preds, n: int = N_SAMPLES) -> None:
    """Sample n test rows, compare prediction vs ground truth, explain them."""
    sample = test_df.sample(n=min(n, len(test_df)), random_state=SEED)
    n_ok = 0
    for idx, row in sample.iterrows():
        true, pred = row[MAIN_TARGET], preds.loc[idx, MAIN_TARGET]
        ok = true == pred
        n_ok += ok
        _, _, words = explain(model, tokenizer, row["raw_text"])
        print(f"\n[{'OK ' if ok else 'ERR'}] {row['raw_text'][:200]}")
        print(f"   true={true} | pred={pred} ({preds.loc[idx, 'confidence']:.2f})")
        shown = [f"{t}: {row[t]}->{preds.loc[idx, t]}" for t in TARGET_COLS[1:]
                 if not (row[t] in DEFAULT_VALUES
                         and preds.loc[idx, t] in DEFAULT_VALUES)]
        print("   communities (true->pred):", ", ".join(shown) or "none")
        print("   key words:", ", ".join(f"{w} ({s:.2f})" for w, s in words))
    print(f"\nSample accuracy ({MAIN_TARGET}): {n_ok}/{len(sample)}")



def save_artifacts(model, tokenizer, classes: dict) -> None:
    """Save weights, tokenizer, encoder config and class names."""
    BERT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), BERT_DIR / "model.pt")
    tokenizer.save_pretrained(BERT_DIR)
    model.encoder.config.save_pretrained(BERT_DIR)
    (BERT_DIR / "classes.json").write_text(json.dumps(classes))


def load_artifacts():
    """Return (model in eval mode, tokenizer, classes) from BERT_DIR."""
    classes = json.loads((BERT_DIR / "classes.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(BERT_DIR))
    model = MultiHeadBert(encoder, {t: len(c) for t, c in classes.items()})
    model.load_state_dict(torch.load(BERT_DIR / "model.pt", map_location=DEVICE))
    return model.to(DEVICE).eval(), tokenizer, classes


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"device: {DEVICE}")

    train_df, test_df = load_data(TRAIN_CSV), load_data(TEST_CSV)
    print(f"train: {len(train_df)} rows | test: {len(test_df)} rows")

    classes = {t: sorted(set(train_df[t]) | set(test_df[t])) for t in TARGET_COLS}
    class_maps = {t: {c: i for i, c in enumerate(cs)} for t, cs in classes.items()}

    fit_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED,
                                      stratify=train_df[MAIN_TARGET])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MultiHeadBert(AutoModel.from_pretrained(MODEL_NAME),
                          {t: len(c) for t, c in classes.items()}).to(DEVICE)

    train_model(model, tokenizer, fit_df, val_df, class_maps)
    save_artifacts(model, tokenizer, classes)

    predictions = evaluate_model(model, tokenizer, test_df, classes)
    inspect_samples(model, tokenizer, test_df, predictions)
