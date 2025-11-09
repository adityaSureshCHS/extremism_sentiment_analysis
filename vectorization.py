import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

# --- VADER sentiment (for the extra feature) ---
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon once (comment out after first run if you want)
nltk.download("vader_lexicon")


# 1. Load the dataset ---------------------------------------------------------

# Change this path to wherever Kaggle saved your CSV
df = pd.read_csv("extremism_dataset.csv")  # e.g. "kaggle/input/your-dataset/file.csv"

# Basic sanity check
expected_cols = {"Original_Message", "Extremism_Label"}
missing_cols = expected_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")


# 2. Clean / encode labels ----------------------------------------------------

def clean_label(raw_label: str) -> str:
    """
    Normalize labels by:
    - Lowercasing
    - Removing spaces and underscores
    """
    if not isinstance(raw_label, str):
        return ""

    cleaned = raw_label.lower()
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace("_", "")
    return cleaned

df["Label_Clean"] = df["Extremism_Label"].apply(clean_label)

# Map to 0/1
label_map = {
    "extremist": 1,
    "nonextremist": 0,   # matches "non_extremist" after cleaning
}

def encode_label(cleaned: str) -> int:
    if cleaned not in label_map:
        raise ValueError(f"Unexpected label after cleaning: {cleaned!r}")
    return label_map[cleaned]

df["Label_Int"] = df["Label_Clean"].apply(encode_label)

# Our final labels
y = df["Label_Int"].values.astype(np.int64)


# 3. Prepare text (input) -----------------------------------------------------

# Fill NaNs with empty strings to avoid issues
texts = df["Original_Message"].fillna("").astype(str).values


# 4. Custom transformer: VADER sentiment -------------------------------------

class VaderSentimentTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that takes an array-like of texts
    and returns a 2D numpy array of shape (n_samples, 1)
    with the VADER compound score for each text.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        # Nothing to learn; this is a stateless transformer.
        return self

    def transform(self, X):
        # X is an iterable of texts
        scores = []
        for text in X:
            s = self.analyzer.polarity_scores(text)
            # Use just the compound score as a single feature
            scores.append([s["compound"]])
        return np.array(scores, dtype=np.float32)


# 5. TF-IDF vectorizer for text ----------------------------------------------

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # unigrams + bigrams
    max_features=3000,    # adjust to 2000/5000 depending on experiments
    min_df=3,             # drop very rare terms
)


# 6. Combine TF-IDF + VADER into a single feature space ----------------------

feature_union = FeatureUnion(
    transformer_list=[
        ("tfidf", tfidf_vectorizer),         # high-dim text features
        ("vader", VaderSentimentTransformer())  # 1D sentiment feature
    ]
)

# Wrap in a Pipeline so you later can just plug in a classifier on top
vectorization_pipeline = Pipeline(
    steps=[
        ("features", feature_union),
        # You could add a scaler for dense features here if you add more
    ]
)


# 7. Fit the vectorizer and transform the whole dataset ----------------------

X = vectorization_pipeline.fit_transform(texts)

print("Feature matrix shape:", X.shape)  # (n_samples, n_features_total)
print("Example labels:", y[:10])

# Optional: train/validation split for later modeling
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)