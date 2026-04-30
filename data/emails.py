"""
data/emails.py
--------------
This file contains our training dataset.

Each email has:
- "text"  → the email content
- "label" → its category

Categories used:
1. Inquiry   → asking questions
2. Complaint → reporting problems
3. Feedback  → giving suggestions/opinions
4. Spam      → unwanted/promotional messages
"""


#this file contains our training dataset of emails with their corresponding labels. Each email is categorized as either an Inquiry, Complaint, Feedback, or Spam. This dataset will be used to train our machine learning model for email classification.
TRAINING_EMAILS = [

    #inquiry
    {
        "text": "Hi, I wanted to ask about pricing plans. Do you offer monthly subscription?",
        "label": "Inquiry"
    },
    {
        "text": "What documents are required to open an account?",
        "label": "Inquiry"
    },
    {
        "text": "Can you explain your services and available packages?",
        "label": "Inquiry"
    },
    {
        "text": "What are your working hours?",
        "label": "Inquiry"
    },


    #complaint
    {
        "text": "I have been waiting for my order for 3 weeks. This is unacceptable.",
        "label": "Complaint"
    },
    {
        "text": "Your support team is not helping me at all.",
        "label": "Complaint"
    },
    {
        "text": "The product I received is damaged. I want a refund.",
        "label": "Complaint"
    },
    {
        "text": "Your app keeps crashing. Very frustrating.",
        "label": "Complaint"
    },


    #feedback
    {
        "text": "The new update looks great. Good job!",
        "label": "Feedback"
    },
    {
        "text": "You should improve onboarding with tutorials.",
        "label": "Feedback"
    },
    {
        "text": "Support team was very helpful. Excellent service.",
        "label": "Feedback"
    },
    {
        "text": "The app is good but font size is too small.",
        "label": "Feedback"
    },


    #spam
    {
        "text": "Congratulations! You won $1000. Click now!",
        "label": "Spam"
    },
    {
        "text": "Earn money from home. No experience needed!",
        "label": "Spam"
    },
    {
        "text": "Urgent: Your account is hacked. Click link now!",
        "label": "Spam"
    },
    {
        "text": "Buy cheap products with huge discount today!",
        "label": "Spam"
    },
]