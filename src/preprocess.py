
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def load_and_preprocess():
    print("Loading datasets...")

    true_df = pd.read_csv("data/True.csv")
    fake_df = pd.read_csv("data/Fake.csv")

    true_df["label"] = 1
    fake_df["label"] = 0

    # Rename text column if needed
    if "text" not in true_df.columns and "title" in true_df.columns:
        true_df["text"] = true_df["title"] + " " + true_df["text"]
    if "text" not in fake_df.columns and "title" in fake_df.columns:
        fake_df["text"] = fake_df["title"] + " " + fake_df["text"]

    print("Cleaning text...")
    true_df["clean_text"] = true_df["text"].apply(clean_text)
    fake_df["clean_text"] = fake_df["text"].apply(clean_text)

    print("Combining datasets...")
    df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

    print("Applying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"]

    return X, y, vectorizer, df

if __name__ == "__main__":
    # paths to your datasets
    true_path = "data/True.csv"
    fake_path = "data/Fake.csv"

    print("Loading datasets...")
    import pandas as pd

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    print("Cleaning text...")

    df_true["clean_text"] = df_true["text"].apply(clean_text)
    df_fake["clean_text"] = df_fake["text"].apply(clean_text)

    print("Saving cleaned data...")

    df_true.to_csv("data/True_clean.csv", index=False)
    df_fake.to_csv("data/Fake_clean.csv", index=False)

    print("DONE! Cleaned files saved as True_clean.csv and Fake_clean.csv")
