import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def preprocess(path):
    df = pd.read_csv(path)
    df["clean_text"] = df["text"].apply(clean_text)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    
    return X, df["label"], vectorizer
