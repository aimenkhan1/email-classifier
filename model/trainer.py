"""
model/trainer.py
----------------
This file trains an Email Classification model and saves it.

Flow:
Email text → cleaning → TF-IDF → Logistic Regression → Prediction
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Our dataset + cleaning function
from data.emails import TRAINING_EMAILS
from utils.preprocessor import clean_text


#Path where model will be saved
MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier.pkl")


#ML pipeline
def build_pipeline():
    """
    We create a pipeline (step-by-step process):
    1. Convert text → numbers (TF-IDF)
    2. Apply Logistic Regression classifier
    """

    pipeline = Pipeline([
        # Converting text into numerical features
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # using single words + pairs of words
            max_features=5000     # limit number of features
        )),

        #Classification model
        ("model", LogisticRegression(
            max_iter=1000         #
        ))
    ])

    return pipeline

#training and saving
def train_and_save():
    """
    This function does full training:
    - load data
    - clean data
    - train model
    - test model
    - save model
    """

    #extracting texts and labels from our dataset
    texts = [item["text"] for item in TRAINING_EMAILS]
    labels = [item["label"] for item in TRAINING_EMAILS]

    #cleaning text 
    texts = [clean_text(t) for t in texts]

    #split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,     
        random_state=42
    )

  
    model = build_pipeline()

    #train
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

#test
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, predictions))

  
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("\nModel saved successfully!")

    return model


#loading
def load_model():
    """
    Load saved model.
    If not found → train new one.
    """

    if not os.path.exists(MODEL_PATH):
        print("Model not found. Training new model...")
        return train_and_save()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully!")
    return model


#main
if __name__ == "__main__":
    train_and_save()