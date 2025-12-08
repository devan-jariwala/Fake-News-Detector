import sys
import pickle
import os
import nltk
import string
from nltk.corpus import stopwords

nltk.download("stopwords")

# -------------------------
# CLEANING FUNCTION
# -------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

# -------------------------
# LOAD MODEL + VECTORIZER
# -------------------------
print("Loading model and vectorizer...")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# UTILITY FUNCTIONS
# -------------------------
def classify_text(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "REAL" if pred == 1 else "FAKE"

def classify_file(path):
    if not os.path.exists(path):
        print("\nERROR: File not found:", path)
        return
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"\nLoaded article from: {path}")
    print("\nRunning prediction...\n")
    result = classify_text(text)
    print("Prediction:", result)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

# -------------------------
# MAIN MENU UI
# -------------------------
def main_menu():
    while True:
        print("\n==============================")
        print("     FAKE NEWS DETECTOR")
        print("==============================")
        print("1. Classify article from file")
        print("2. Paste article manually")
        print("3. Clear screen")
        print("4. Exit")
        print("==============================")

        choice = input("Choose an option (1-4): ")

        # Option 1: file article
        if choice == "1":
            path = input("\nEnter file path (e.g., articles/test1.txt): ")
            classify_file(path)

        # Option 2: manual paste
        elif choice == "2":
            print("\nPaste article below (finish with ENTER):")
            article = input("> ")
            print("\nPrediction:", classify_text(article))

        # Option 3: clear terminal
        elif choice == "3":
            clear_screen()

        # Exit
        elif choice == "4":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Try again.")

if __name__ == "__main__":
    main_menu()
