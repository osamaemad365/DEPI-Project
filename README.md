# 🎭 Customer Sentiment Analysis

A machine learning pipeline that classifies customer tweets into **positive**, **negative**, or **neutral** sentiment — with experiment tracking via MLflow and an interactive Streamlit web app.

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Multi-class text classification (3 sentiment classes) |
| **Dataset** | ~27,000 tweets with demographic metadata |
| **Models** | Logistic Regression · Random Forest |
| **Tracking** | MLflow |
| **App** | Streamlit (single tweet + batch CSV upload) |

---

## 📂 Project Structure

```
customer-sentiment-analysis/
│
├── data/
│   ├── train.csv               # Raw training data
│   ├── test.csv                # Raw test data
│   ├── cleaned_train.csv       # Preprocessed training data
│   └── cleaned_test.csv        # Preprocessed test data
│
├── models/                     # Saved model artifacts (generated after running notebook)
│   ├── sentiment_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── standard_scaler.joblib
│
├── project_3.ipynb             # Main analysis notebook
├── app.py                      # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

Each record contains a tweet and the following metadata:

| Column | Description |
|---|---|
| `textID` | Unique tweet identifier |
| `text` | Original tweet content |
| `selected_text` | Key phrase from the tweet |
| `sentiment` | Target label: `positive` / `negative` / `neutral` |
| `Age of User` | Age group (`0-20`, `21-30`, `31-45`, `46-60`, `60-70`) |
| `Time of Tweet` | Time of day (`morning`, `noon`, `night`) |
| `Country` | User's country |

**Class distribution:**

| Sentiment | Count | Share |
|---|---|---|
| Neutral | 11,118 | 40.5% |
| Positive | 8,582 | 31.2% |
| Negative | 7,781 | 28.3% |

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/customer-sentiment-analysis.git
cd customer-sentiment-analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the notebook

Open `project_3.ipynb` in Jupyter or VS Code and run all cells top-to-bottom. The notebook will:

1. Load and explore the raw data
2. Preprocess text (clean tweets → `cleaned_text`)
3. Train Logistic Regression and Random Forest models
4. Log experiments to MLflow
5. Save model artifacts to `models/`

### View MLflow dashboard

```bash
mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) to compare runs.

### Launch the Streamlit app

```bash
streamlit run app.py
```

The app supports:
- **Single tweet** analysis with confidence scores
- **Batch CSV upload** — upload any CSV with a `text` column and download predictions

---

## 🧹 Text Preprocessing

Each tweet goes through the following cleaning steps:

1. Lowercase conversion
2. Remove URLs (`http://...`)
3. Remove `@mentions` and `#hashtags`
4. Remove non-ASCII characters
5. Remove punctuation and digits
6. Normalize whitespace

---

## 📈 Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | ~75% | ~0.74 |
| Random Forest | ~72% | ~0.71 |

> Exact metrics vary by run. Check MLflow for the latest experiment results.

---

## 🔭 Future Work

- [ ] Fine-tune a pre-trained transformer (BERT / RoBERTa) for higher accuracy
- [ ] Add SHAP explanations for model interpretability
- [ ] Incorporate demographic features (age, country, time of tweet)
- [ ] Deploy the Streamlit app to Streamlit Community Cloud

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![MLflow](https://img.shields.io/badge/MLflow-2.10-red?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-ff4b4b?logo=streamlit)

---

## 📄 License

MIT License — feel free to use and adapt.
