import re
from pathlib import Path

import pandas as pd


# =========================================================
# DATASET PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAKE_NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "News-_dataset"
    / "Fake.csv"
)

REAL_NEWS_PATH = (
    PROJECT_ROOT
    / "data"
    / "News-_dataset"
    / "True.csv"
)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Clean news text for machine-learning processing.

    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove HTML-like tags
    - Remove non-letter characters
    - Remove extra whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep alphabetic characters and spaces
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# LOAD DATASET
# =========================================================

def load_news_dataset():
    """
    Load the ISOT Fake News Dataset.

    Label convention:
        0 = Real
        1 = Fake
    """

    if not FAKE_NEWS_PATH.exists():
        raise FileNotFoundError(
            f"Fake.csv not found at: {FAKE_NEWS_PATH}"
        )

    if not REAL_NEWS_PATH.exists():
        raise FileNotFoundError(
            f"True.csv not found at: {REAL_NEWS_PATH}"
        )

    fake_df = pd.read_csv(
        FAKE_NEWS_PATH
    )

    real_df = pd.read_csv(
        REAL_NEWS_PATH
    )

    # Assign labels
    fake_df["label"] = 1
    real_df["label"] = 0

    # Combine datasets
    df = pd.concat(
        [real_df, fake_df],
        ignore_index=True
    )

    # Ensure required text columns exist
    required_columns = [
        "title",
        "text"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' "
                "was not found in the dataset."
            )

    # Replace missing text
    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    # Combine title + article body
    df["content"] = (
        df["title"]
        + " "
        + df["text"]
    )

    # Clean combined content
    df["clean_text"] = (
        df["content"]
        .apply(clean_text)
    )

    # Remove empty rows
    df = df[
        df["clean_text"].str.len() > 0
    ].copy()

    # Remove exact duplicates
    df = df.drop_duplicates(
        subset=["clean_text"]
    ).reset_index(drop=True)

    # Shuffle dataset
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df


# =========================================================
# DATASET SUMMARY
# =========================================================

def get_dataset_summary(df):
    """
    Return useful dataset statistics.
    """

    total_records = len(df)

    real_count = int(
        (df["label"] == 0).sum()
    )

    fake_count = int(
        (df["label"] == 1).sum()
    )

    return {
        "total_records": total_records,
        "real_articles": real_count,
        "fake_articles": fake_count
    }


# =========================================================
# TEST SCRIPT
# =========================================================

if __name__ == "__main__":

    dataset = load_news_dataset()

    summary = get_dataset_summary(
        dataset
    )

    print(
        "\nTruthLens AI Dataset Loaded Successfully"
    )

    print(
        "-" * 45
    )

    print(
        f"Total records : "
        f"{summary['total_records']}"
    )

    print(
        f"Real articles : "
        f"{summary['real_articles']}"
    )

    print(
        f"Fake articles : "
        f"{summary['fake_articles']}"
    )

    print(
        "\nSample cleaned text:\n"
    )

    print(
        dataset[
            "clean_text"
        ].iloc[0][:500]
    )