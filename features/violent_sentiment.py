#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def load_data(path):
    # Read TSV, strip header whitespace, normalize to snake_case
    df = (
        pd.read_csv(path, sep='\t', skipinitialspace=True)
          .rename(columns=lambda c: c.strip().lower().replace(' ', '_'))
          .dropna(subset=['violence'])
    )
    df['label'] = df['violence'].map({'Yes': 1, 'No': 0})
    return df['posttext'], df['label']

def make_pipeline(model):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10_000, ngram_range=(1,2))),
        ('clf',  model)
    ])

def main():
    X, y = load_data('data/violence_dataset.tdf')
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        'SVM': LinearSVC(),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, mdl in models.items():
        pipe = make_pipeline(mdl)
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xval)
        print(f"\n=== {name} Classification Report ===")
        print(classification_report(yval, preds, digits=4))

if __name__ == '__main__':
    main()