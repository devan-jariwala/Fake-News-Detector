import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
import pickle

print("Loading cleaned combined dataset...")
df = pd.read_csv("data/combined_clean.csv")

# Drop any empty rows that might still exist
df = df.dropna(subset=["clean_text", "label"])
df = df[df["clean_text"].str.strip() != ""]

# Shuffle the dataset
df = shuffle(df, random_state=42).reset_index(drop=True)

print(f"Dataset size: {len(df)}")
print(df["label"].value_counts())

print("\nVectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),   # BIG improvement for news text
    min_df=2             # removes rare noise words
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("Splitting into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=5000, n_jobs=-1)
model.fit(X_train, y_train)

print("Evaluating...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("Accuracy:", accuracy_score(y_test, preds))

print("\nSaving model + vectorizer...")
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("DONE! New model saved in models/.")
