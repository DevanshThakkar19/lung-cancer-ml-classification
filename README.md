# Lung Cancer Risk Classification

Survey-based lung cancer prediction (YES/NO) under severe class imbalance — CS6140 Machine Learning midterm project.

## Overview

| | |
|--|--|
| **Problem** | Binary classification from survey features |
| **Challenge** | ~6.9:1 class imbalance |
| **Stack** | Python, pandas, scikit-learn, imbalanced-learn |
| **Techniques** | EDA, RandomOverSampler/SMOTE, SelectKBest, PCA, RFE, Grid Search CV, Random Forest vs SVM |
| **Metrics** | Holdout F1 / precision / recall (not accuracy alone) |

## Dataset

Kaggle: [Survey Lung Cancer](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)

The CSV is already included at `data/survey_lung_cancer.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook lung_cancer_prediction.ipynb
```

Or with JupyterLab:

```bash
jupyter lab lung_cancer_prediction.ipynb
```

## Key result

Early models looked strong on accuracy while mostly predicting the majority class. After rebalancing and evaluating F1/recall on holdout data, minority-class performance improved materially — a concrete example of catching a misleading metric before shipping conclusions.

## Author

**Devansh Thakkar** — MS Artificial Intelligence, Northeastern University  
[github.com/DevanshThakkar19](https://github.com/DevanshThakkar19)
