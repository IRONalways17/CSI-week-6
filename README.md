# Machine Learning Model Comparison & Hyperparameter Tuning

This repository contains:

1. **analysis.ipynb** &mdash; A fully executable Jupyter/Colab notebook that:
   * Loads the UCI Wine dataset (also included as *wine.csv* for convenience).
   * Trains three baseline classifiers (Logistic Regression, Random Forest, SVC).
   * Evaluates each with accuracy, precision, recall & F1-score.
   * Performs hyperparameter tuning:
       * **GridSearchCV** on Random Forest.
       * **RandomizedSearchCV** on SVC.
   * Compares tuned models and selects the best by weighted F1.
   * Saves the best model as a Joblib artifact.

2. **wine.csv** &mdash; The dataset used by the notebook (178 × 14, no preprocessing required).

3. **README.md** &mdash; quick-start instructions.

---

## Quick Start

```bash
# Clone repository and enter it
$ git clone <repo-url>  && cd <repo>

# (Optional) create virtual environment
$ python -m venv venv && source venv/bin/activate

# Install requirements
$ pip install -r requirements.txt

# Launch the notebook locally
$ jupyter notebook analysis.ipynb
```

*Or open *analysis.ipynb* directly in **Google Colab** – everything is self-contained.*

---

## Repository Structure

```
├── analysis.ipynb   # Colab/ Jupyter notebook with full workflow
├── wine.csv         # Dataset (UCI Wine)
├── README.md        # Project overview & instructions
└── requirements.txt # Minimal Python deps (sklearn, pandas, scipy, joblib)
```

---

## Requirements

* Python ≥3.8
* pandas, scikit-learn, scipy, joblib (see *requirements.txt*)

---

## Results Snapshot

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|---------:|----------:|-------:|---------:|
| LogisticRegression | ~0.97 | 0.97 | 0.97 | 0.97 |
| RandomForest (tuned) | **1.00** | **1.00** | **1.00** | **1.00** |
| SVC (tuned)        | 0.99 | 0.99 | 0.99 | 0.99 |

*Exact numbers may vary slightly by random state.*

---
#Aaryan Choudhary
