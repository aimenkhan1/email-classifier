# ✉️ Email Classification System


A machine learning system that classifies incoming emails into one of four categories: **Inquiry**, **Complaint**, **Feedback**, or **Spam** — complete with confidence scores and a clean web UI.

---

## 📸 What It Does

Paste any email into the interface and the system will:
- Predict the most likely category
- Show a confidence percentage
- Display a breakdown of probability scores across all four classes
- Show you the cleaned text that was passed to the ML model

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Logistic Regression (scikit-learn) |
| Text Features | TF-IDF Vectorizer (unigrams + bigrams) |
| Preprocessing | Custom text cleaner (regex, punctuation removal) |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 📂 Project Structure

```
email-classifier/
│
├── app.py                  # Streamlit UI — entry point
│
├── data/
│   └── emails.py           # 48 labeled training emails across 4 categories
│
├── model/
│   ├── trainer.py          # Builds and trains the sklearn pipeline, saves model to disk
│   ├── predictor.py        # Loads model, runs predictions, returns structured results
│   └── classifier.pkl      # Saved trained model (auto-generated on first run)
│
├── utils/
│   └── preprocessor.py     # Text cleaning: lowercase, remove URLs, strip punctuation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ ML Pipeline

```
Raw email text
     ↓
Text Preprocessor     → lowercase, remove URLs + punctuation, collapse spaces
     ↓
TF-IDF Vectorizer     → 5000 features, unigrams + bigrams, sublinear TF scaling
     ↓
Logistic Regression   → multi-class classifier (C=1.0, max_iter=1000)
     ↓
Predicted Label + Confidence Scores for all 4 classes
```

---

## 🏷 Categories

| Category | Description |
|---|---|
| 🔍 Inquiry | Questions, information requests, pre-sales queries |
| ⚠️ Complaint | Problems, dissatisfaction, refund/replacement requests |
| 💬 Feedback | Suggestions, compliments, improvement ideas |
| 🚫 Spam | Promotional, phishing, or unsolicited messages |

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/email-classifier.git
cd email-classifier
```

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

```

### 3. (Optional) Retrain the model manually
```bash
python model/trainer.py
```
> The model also trains automatically on the first app run if no saved model is found.

### 4. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Requirements

```
streamlit==1.35.0
scikit-learn==1.4.2
```

Python 3.10 or higher recommended.

---

## 🧪 Example Inputs

**Inquiry:**
> "Hi, I'd like to know what pricing plans you offer and whether there's a free trial."

**Complaint:**
> "I placed an order 10 days ago and it still hasn't arrived. This is unacceptable."

**Feedback:**
> "The new dashboard is clean and easy to use. Would love an export to PDF option."

**Spam:**
> "CONGRATULATIONS! You've been selected to WIN a FREE iPhone! Click here NOW!"

---

## 📝 Notes

- Training data is in `data/emails.py` — easy to extend with more examples per category
- The model auto-saves to `model/classifier.pkl` after training
- All preprocessing is in `utils/preprocessor.py` — same function is used for both training and prediction to avoid any data leakage issues

---

*Built as part of the Calder AI/ML Internship selection process.*
