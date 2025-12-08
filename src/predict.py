import pickle
import sys

# Load model + vectorizer
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Check if a file was provided
if len(sys.argv) < 2:
    print("Usage: python src/predict.py articles/test1.txt")
    exit()

article_path = sys.argv[1]

# Read the article
with open(article_path, "r", encoding="utf-8") as f:
    article = f.read()

# Vectorize and predict
X = vectorizer.transform([article])
prediction = model.predict(X)[0]

label = "REAL" if prediction == 1 else "FAKE"
print("\nPrediction:", label)
