# 🚀 Lead Scoring & Similarity Ranking App

This web application allows users to **upload a CSV of Growjo dataset**, compute **lead scores** using a machine learning model, and optionally **rank companies based on keyword similarity** (e.g., "AI machine learning"). It's built using **Flask**, with support for filtering and pagination.

---
Lead Score – How Good Is the Company?
What it is: A percentage (0–100%) that shows how strong or valuable a company is as a business lead.

How it works: A machine learning model looks at company data (like funding, size, growth) and gives each company a score.

Why it helps: It helps you focus on top companies first instead of wasting time on low-potential ones.

Similarity Score – How Relevant Is the Company?
What it is: A score (0–1) that shows how closely a company matches your keyword or interest.

How it works: It compares your search term (like “AI marketing”) to each company’s description using AI and text comparison.

Why it helps: You find companies that are doing exactly what you’re looking for, even if they are labeled differently.
## 🔍 Features

- 📊 **Lead Scoring**: Predicts potential value of each company using a trained ML model (`lead_score_model.pkl`).
- 🧠 **Keyword Similarity Ranking**: Uses cosine similarity between a keyword vector and company vectors (`simscore_model.pkl`).
- 🎯 **Filters**: Filter companies by `country`, `city`, and `industry`.
- 📃 **Pagination**: Clean UI with page-wise navigation for large datasets.
- 📁 Upload CSV directly from browser.

---

## 🧠 Tech Stack

- **Backend**: Flask, scikit-learn, joblib
- **Frontend**: Bootstrap, Jinja2 (via Flask templates)
- **Data**: Pandas, NumPy, Description vector CSV
- **ML Models**: Pickle/joblib models for lead scoring and similarity

---

## 📁 Project Structure


growjo/
├── app.py # Flask application
├── preprocessor.py # Data cleaning & transformation
├── keyword_similarity.py # Keyword similarity logic
├── lead_score_model.pkl # Lead scoring ML model
├── simscore_model.pkl # Vectorizer model for keyword similarity
├── Description.csv # Precomputed company description vectors
├── requirements.txt
├── templates/
│ ├── index.html # Upload form
│ └── results.html # Results with filters & pagination
└── static/ 
STEPS-

1)install dependencies

pip install -r requirements.txt

2)run the flask app

python app.py

✍️ Author
Debrup Ghosh
