Fake News Detection System

This project builds a full end to end pipeline that detects whether a news article is real or fake using classic data-management techniques and machine learning models. It was designed for the CS 210: Data Management in Data Science final project, where the goal is to combine data collection, cleaning, transformation, storage, modeling, and evaluation into one reproducible system.

Project Overview

The system focuses on two major ideas:

1. Text-based fake-news classification:
Using TF-IDF features and Logistic Regression to detect writing-style differences between real and fake news.

2. Article logic checking:
A second stage that uses a lightweight transformer model to evaluate whether the article’s content itself appears plausible or fictional.

Features

- Full preprocessing pipeline (cleaning, tokenization, stopwords, etc.)

- Merges two datasets (True.csv + Fake.csv) into one labeled dataset

- TF-IDF vectorization with a saved vocabulary

- Logistic Regression classifier for baseline predictions

- DistilBERT-based factuality checker (optional extension)

- SQLite database for saving predictions

Ability to classify:

- User-typed articles

- Full articles stored in .txt files (batch mode)

- Modular code structure inside the src/ folder

Repository Structure...
Fake-News-Detector/
│
├── data/                    # Raw datasets (True.csv, Fake.csv)
├── models/                  # Saved logistic regression model & vectorizer
├── articles/                # Test articles for evaluation
├── src/
│   ├── preprocess.py        # Cleans text + builds combined_clean.csv
│   ├── train.py             # Trains LR model and saves model.pkl
│   ├── pipeline.py          # Loads model and classifies articles
│   ├── predict.py           # Classify a .txt file in one command
│   ├── database.py          # Saves predictions into SQLite
│   └── logic_checker.py     # Optional transformer-based checker
├── predictions.db           # Database storing predictions
└── README.md                # Project documentation

How to Run the Project
1. Create the environment
conda create -n fake_news python=3.11
conda activate fake_news
pip install -r requirements.txt

2. Preprocess the dataset
python src/preprocess.py


This generates a cleaned dataset inside data/.

3. Train the model
python src/train.py


This creates:

models/model.pkl
models/vectorizer.pkl

4. Predict a full article from a file

Put your article in:

articles/test1.txt


Then run:

python src/pipeline.py articles/test1.txt


You will get:

Prediction: REAL or FAKE

Why This Project Matters:

Fake news spreads rapidly online and is often hard to identify manually.
This project attempts to:

- Demonstrate structured data pipelines

- Practice text cleaning, feature engineering, and dataset integration

- Compare stylistic vs factual detection

- Build a reproducible ML workflow

- Explore the limitations of classic ML vs. modern NLP models

It ties directly into data-management topics from CS 210:

- Pandas: Used throughout preprocessing to load datasets, merge them, clean text columns, handle missing values, and export processed files. This directly reflects the course focus on data manipulation and working with real-world messy data.

- Normalization & cleaning: You built a full text-cleaning pipeline that standardizes case, removes punctuation, drops stop words, and handles invalid rows, which mirrors the class content on cleaning, transforming, and normalizing data before analysis.

- Feature extraction: Generating TF-IDF vectors from text applies the course idea of converting raw data into structured features that can be modeled, demonstrating practical feature engineering.

- Indexing & storage: You used a directory structure, stored datasets in /data, and saved models + vectorizers for later inference. You also prepared (and optionally implemented) SQLite to index predictions, connecting directly to the database module of the course.

- Modeling & evaluation: Logistic regression, train/test splitting, accuracy, precision, recall, and F1-score all directly reflect the class focus on building statistical models, evaluating them, and understanding performance metrics.

- Semantics: Your extended logic-checker and fake-news classifier demonstrate how meaning, context, and interpretation matter in data management — aligning with the course’s emphasis on understanding the semantic structure of text data and how models represent meaning.

Future Extensions: 

- Build a web or mobile UI for article classification

- Add article source credibility scoring

- Add topic detection using LDA

- Use a fine-tuned BERT model for higher accuracy

- Evaluate classification drift over time

Team Members:

- Devan Jariwala

- Rajvi Maniar

- Hirav Shah

- Devesh Tyavanagimatt
