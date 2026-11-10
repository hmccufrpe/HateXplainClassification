import re
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

TRAIN_CSV = "../data/train.csv"        
TEST_CSV = "../data/test.csv"
MODEL_PATH = "nb_models.joblib"
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
USE_COMPLEMENT_NB = True  


def clean_text(text: str) -> str:
    """Lowercase, strip placeholders/URLs/non-letters and collapse spaces."""
    text = str(text).lower()
    text = text.replace("<number>", " ")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline() -> GridSearchCV:
    """TF-IDF (uni+bigrams) + Naive Bayes, with alpha tuned by macro-F1."""
    nb = ComplementNB() if USE_COMPLEMENT_NB else MultinomialNB()
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True)),
        ("nb", nb),
    ])
    grid = {"nb__alpha": [0.1, 0.3, 0.5, 1.0]}
    return GridSearchCV(pipe, grid, scoring="f1_macro", cv=3, n_jobs=-1)


def load_data(path: str) -> pd.DataFrame:
    """Load a split CSV, keep the original text, add a cleaned column."""
    df = pd.read_csv(path, keep_default_na=False)
    df[TARGET_COLS] = df[TARGET_COLS].replace("", "None")  # NaNs from older splits
    df["raw_text"] = df[TEXT_COL].astype(str)
    df[TEXT_COL] = df["raw_text"].map(clean_text)
    return df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)


def train_models(train_df: pd.DataFrame) -> dict:
    """Train one model per target on the training split only."""
    models = {}
    for target in TARGET_COLS:
        search = build_pipeline()
        search.fit(train_df[TEXT_COL], train_df[target])
        models[target] = search.best_estimator_
        print(f"[train] {target}: best alpha = {search.best_params_['nb__alpha']}")
    return models


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


def evaluate_models(models: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate every model on the held-out test split; return predictions."""
    preds = pd.DataFrame(index=test_df.index)
    for target, model in models.items():
        preds[target] = model.predict(test_df[TEXT_COL])
        print(f"\n===== {target} =====")
        print(f"accuracy: {accuracy_score(test_df[target], preds[target]):.4f} | "
              f"macro-F1: {f1_score(test_df[target], preds[target], average='macro'):.4f}")
        print(classification_report(test_df[target], preds[target],
                                    zero_division=0))
        save_confusion_matrix(test_df[target], preds[target], target)
    return preds


def explain(model: Pipeline, text: str, k: int = TOP_K_WORDS):
    """Local XAI: predicted class + words that most favour it over runner-up.

    ComplementNB stores inverted statistics, so the sign is flipped to keep
    "positive score = supports the predicted class".
    """
    tfidf, nb = model.named_steps["tfidf"], model.named_steps["nb"]
    cleaned = clean_text(text)
    x = tfidf.transform([cleaned])
    probs = nb.predict_proba(x)[0]
    order = np.argsort(probs)[::-1]
    pred_idx, runner_idx = order[0], order[1]

    sign = -1.0 if isinstance(nb, ComplementNB) else 1.0
    diff = sign * (nb.feature_log_prob_[pred_idx] - nb.feature_log_prob_[runner_idx])
    contrib = x.multiply(diff).tocoo()
    vocab = tfidf.get_feature_names_out()
    words = sorted(((vocab[j], float(v)) for j, v in zip(contrib.col, contrib.data)),
                   key=lambda t: t[1], reverse=True)[:k]
    return nb.classes_[pred_idx], probs[pred_idx], words


def inspect_samples(models, test_df, preds, n=N_SAMPLES):
    """Sample n test rows, compare prediction vs ground truth, explain them."""
    sample = test_df.sample(n=min(n, len(test_df)), random_state=SEED)
    n_ok = 0
    for idx, row in sample.iterrows():
        true, pred = row[MAIN_TARGET], preds.loc[idx, MAIN_TARGET]
        ok = true == pred
        n_ok += ok
        _, prob, words = explain(models[MAIN_TARGET], row["raw_text"])
        print(f"\n[{'OK ' if ok else 'ERR'}] {row['raw_text'][:200]}")
        print(f"   true={true} | pred={pred} ({prob:.2f})")
        shown = [f"{t}: {row[t]}->{preds.loc[idx, t]}" for t in TARGET_COLS[1:]
                 if not (row[t] in DEFAULT_VALUES
                         and preds.loc[idx, t] in DEFAULT_VALUES)]
        print("   communities (true->pred):", ", ".join(shown) or "none")
        print("   key words:", ", ".join(f"{w} ({s:.2f})" for w, s in words))
    print(f"\nSample accuracy ({MAIN_TARGET}): {n_ok}/{len(sample)}")


if __name__ == "__main__":
    train_df = load_data(TRAIN_CSV)
    test_df = load_data(TEST_CSV)
    print(f"train: {len(train_df)} rows | test: {len(test_df)} rows")
    print("Train class distribution:\n", train_df[MAIN_TARGET].value_counts(), "\n")

    models = train_models(train_df)
    joblib.dump(models, MODEL_PATH)

    predictions = evaluate_models(models, test_df)
    inspect_samples(models, test_df, predictions)
