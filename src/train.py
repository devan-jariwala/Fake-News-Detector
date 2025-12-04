import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

print("Loading cleaned datasets...")
true_df = pd.read_csv("data/True_clean.csv")
fake_df = pd.read_csv("data/Fake_clean.csv")

# Add labels
true_df["label"] = 1
fake_df["label"] = 0

# Combine into one dataset
df = pd.concat([true_df, fake_df], ignore_index=True)

print("Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Logistic Regression...")
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Evaluating model...")
preds = model.predict(X_test)
print(classification_report(y_test, preds))

print("Saving model and vectorizer...")
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("DONE! Model saved in the models/ folder.")