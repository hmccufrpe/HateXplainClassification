import sys
import joblib
from hatexplain_naive_bayes import MAIN_TARGET, MODEL_PATH, TARGET_COLS, clean_text, explain


PHRASE = "write the phrase to classify here"


def main(text: str) -> None:
    """Print label, probabilities, communities and influential words."""
    models = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)

    label, prob, words = explain(models[MAIN_TARGET], text)
    print(f"Phrase: {text}")
    print(f"\n{MAIN_TARGET}: {label} ({prob:.2%})")

    nb = models[MAIN_TARGET].named_steps["nb"]
    probs = models[MAIN_TARGET].predict_proba([cleaned])[0]
    for cls, p in sorted(zip(nb.classes_, probs), key=lambda t: -t[1]):
        print(f"  {cls}: {p:.2%}")

    print("\nCommunities:")
    for target in TARGET_COLS[1:]:
        print(f"  {target}: {models[target].predict([cleaned])[0]}")

    print("\nInfluential words:")
    for word, score in words:
        print(f"  {word}: {score:.3f}")


if __name__ == "__main__":
    main(" ".join(sys.argv[1:]) or PHRASE)
