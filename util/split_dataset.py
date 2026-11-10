import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "../data/hatexplain.csv"
TRAIN_CSV = "../data/train.csv"
TEST_CSV = "../data/test.csv"
TEXT_COL = "comment"
STRATIFY_COL = "label"
TEST_SIZE = 0.2
SEED = 42


def main() -> None:
    """Drop empty comments, split stratified by label and save both files."""
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    df = df[df[TEXT_COL].str.strip() != ""]
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[STRATIFY_COL])

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"train: {len(train_df)} rows -> {TRAIN_CSV}")
    print(f"test:  {len(test_df)} rows -> {TEST_CSV}")
    print("\nTrain distribution:\n", train_df[STRATIFY_COL].value_counts(normalize=True))
    print("\nTest distribution:\n", test_df[STRATIFY_COL].value_counts(normalize=True))


if __name__ == "__main__":
    main()
