"""
app.py
------
This is the main UI of the Email Classification System.

We use Streamlit to build a simple web app where user:
1. Pastes email text
2. Clicks "Classify"
3. Gets predicted category + confidence

...
- All ML logic is in predictor.py
- This file only handles UI (frontend layer)
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model.predictor import predict, CATEGORY_META


#config
st.set_page_config(
    page_title="Email Classifier",
    page_icon="✉️",
    layout="centered"
)


#title
st.title("📧 Email Classification System")
st.write("Paste an email below and the model will classify it.")


#example emails
EXAMPLES = {
    "Inquiry": "Hi, what are your pricing plans?",
    "Complaint": "I am unhappy with your service. Very bad experience.",
    "Feedback": "The app is good but needs improvement in UI.",
    "Spam": "Congratulations! You won a free iPhone!"
}

st.write("### Quick Examples")

cols = st.columns(4)

# Load example into textbox when clicked
for i, (label, text) in enumerate(EXAMPLES.items()):
    if cols[i].button(label):
        st.session_state["input_text"] = text

#input text area
email_text = st.text_area(
    "Enter Email",
    height=180,
    key="input_text"
)


#predict button
if st.button("Classify Email"):

    if not email_text.strip():
        st.warning("Please enter email text.")
    else:
      
        result = predict(email_text)

    
        if "error" in result:
            st.error(result["error"])

        else:
            label = result["predicted_label"]
            confidence = result["confidence"]
            meta = result["meta"]

            st.subheader("Prediction Result")

            st.markdown(f"### {meta.get('emoji','')} {label}")
            st.write(meta.get("description", ""))

            st.success(f"Confidence: {confidence}%")

            # ── Show all scores ──
            st.write("### All Category Scores")

            for cat, score in result["all_scores"].items():
                st.write(f"{cat}: {score}%")


#info section
with st.expander("How it works"):
    st.write("""
    1. Email text is cleaned (remove noise like URLs, punctuation)
    2. TF-IDF converts text into numbers
    3. Logistic Regression predicts category
    4. Model returns confidence scores
    """)


#footer
st.markdown("---")
st.caption("Email Classification System | ML Project")