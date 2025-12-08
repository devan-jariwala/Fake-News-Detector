import pandas as pd
import pickle
from sklearn.metrics import classification_report

print("Loading cleaned datasets...")
true_df = pd.read_csv("data/True_clean.csv")
fake_df = pd.read_csv("data/Fake_clean.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

# FIX FOR NaN CLEAN TEXT
df["clean_text"] = df["clean_text"].fillna("")

print("Loading saved model and vectorizer...")
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Vectorizing cleaned text...")
X = vectorizer.transform(df["clean_text"])
y = df["label"]

print("Evaluating model on FULL dataset:")
preds = model.predict(X)
print(classification_report(y, preds))
