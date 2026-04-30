"""
model/predictor.py
------------------
This file is used to predict the category of a new email.

It takes raw email text → cleans it → sends it to model → returns result.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.trainer import load_model
from utils.preprocessor import clean_text

#UI metadata for each category
CATEGORY_META = {
    "Inquiry": {
        "emoji": "🔍",
        "color": "blue",
        "description": "User is asking a question."
    },
    "Complaint": {
        "emoji": "⚠️",
        "color": "red",
        "description": "User is unhappy or reporting issue."
    },
    "Feedback": {
        "emoji": "💬",
        "color": "green",
        "description": "User is giving opinion or suggestion."
    },
    "Spam": {
        "emoji": "🚫",
        "color": "orange",
        "description": "Unwanted or promotional message."
    }
}


#load model once
_model = None

def get_model():
    """
    Load model only once (not every time).
    Improves performance.
    """
    global _model

    if _model is None:
        _model = load_model()

    return _model


#pred func
def predict(email_text):
    """
    Input:  raw email text
    Output: prediction result (dictionary)
    """

    if not email_text or not email_text.strip():
        return {"error": "Email cannot be empty"}

    model = get_model()

    cleaned_text = clean_text(email_text)

    predicted_label = model.predict([cleaned_text])[0]

    probs = model.predict_proba([cleaned_text])[0]
    classes = model.classes_

    scores = dict(zip(classes, probs))

    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    confidence = scores[predicted_label]

    confidence = round(float(confidence) * 100, 1)
    scores = {k: round(float(v) * 100, 1) for k, v in scores.items()}

  
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "all_scores": scores,
        "meta": CATEGORY_META.get(predicted_label, {})
    }