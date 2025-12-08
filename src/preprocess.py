import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure NLTK resources are available
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Take raw text and return a cleaned version:
    - lowercase
    - remove punctuation
    - remove stopwords
    - lemmatize
    """
    if not isinstance(text, str):
        return ""

    # lowercasing
    text = text.lower()

    # remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # tokenize on spaces
    words = text.split()

    # remove stopwords + lemmatize
    cleaned_words = []
    for w in words:
        if w in STOPWORDS:
            continue
        lemma = LEMMATIZER.lemmatize(w)
        cleaned_words.append(lemma)

    return " ".join(cleaned_words)


def build_clean_dataset(
    true_path: str = "data/True.csv",
    fake_path: str = "data/Fake.csv",
    out_path: str = "data/combined_clean.csv",
) -> None:
    """
    Load True/Fake CSV files, clean text, and save a combined dataset
    with columns: ['original_text', 'clean_text', 'label'].
    label: 1 = real (True.csv), 0 = fake (Fake.csv)
    """
    print("Loading datasets...")
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # We expect at least a 'text' column, and often a 'title' column.
    # If there is a title, concatenate it to the text to give more context.
    def build_text_column(df: pd.DataFrame) -> pd.Series:
        # Prefer 'text' if it exists
        if "text" in df.columns:
            base = df["text"].astype(str)
        else:
            # Try a few common alternatives
            for col in ["content", "article", "body"]:
                if col in df.columns:
                    base = df[col].astype(str)
                    break
            else:
                raise ValueError(
                    "No text-like column found. Expected one of: 'text', 'content', 'article', 'body'."
                )

        # If title exists, prepend it
        if "title" in df.columns:
            title = df["title"].fillna("").astype(str)
            return title + " " + base
        else:
            return base

    print("Building unified text column...")
    true_text = build_text_column(true_df)
    fake_text = build_text_column(fake_df)

    # Build unified dataframes with just what we need
    true_clean = pd.DataFrame(
        {
            "original_text": true_text,
            "label": 1,  # real
        }
    )
    fake_clean = pd.DataFrame(
        {
            "original_text": fake_text,
            "label": 0,  # fake
        }
    )

    combined = pd.concat([true_clean, fake_clean], ignore_index=True)

    print("Cleaning text (this may take a bit)...")
    combined["clean_text"] = combined["original_text"].apply(clean_text)

    # Drop any rows where clean_text ended up empty
    before = len(combined)
    combined = combined[combined["clean_text"].str.strip() != ""]
    combined = combined.dropna(subset=["clean_text"])
    after = len(combined)
    print(f"Dropped {before - after} empty/invalid rows after cleaning.")

    # Shuffle so train/test later is well-mixed
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Saving cleaned combined dataset to {out_path} ...")
    combined.to_csv(out_path, index=False)
    print("DONE: combined_clean.csv written.")


if __name__ == "__main__":
    build_clean_dataset()
