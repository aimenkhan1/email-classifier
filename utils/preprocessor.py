"""
utils/preprocessor.py
---------------------
This file is responsible for cleaning email text before sending it to the model.


Raw text contains noise (URLs, punctuation, etc.) which can confuse the model.
So we clean it first.
"""

import re
import string


#clean single text
def clean_text(text):
    """
    This function takes raw email text and returns cleaned text.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove punctuation
    5. Remove extra spaces
    """

    #step 1:Converting all text to lowercase
    text = text.lower()

    #step 2:Removing URLs (http, https, www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    #step 3:Removing email addresses
    text = re.sub(r"\S+@\S+", "", text)

    #step 4:Removing punctuation marks
    text = text.translate(str.maketrans("", "", string.punctuation))

    #step 5:Removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


#clean batch of texts
def preprocess_batch(texts):
    """
    This function cleans multiple emails at once.

    Input:  list of texts
    Output: list of cleaned texts
    """

    cleaned_list = []

    for t in texts:
        cleaned_list.append(clean_text(t))

    return cleaned_list