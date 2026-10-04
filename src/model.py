from pathlib import Path

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split


# =========================================================
# IMPORT PREPROCESSING FUNCTIONS
# =========================================================

# Supports both:
# python -m src.model
# and, when needed, direct execution from src/
try:
    from src.preprocess import (
        clean_text,
        load_news_dataset
    )
except ModuleNotFoundError:
    from preprocess import (
        clean_text,
        load_news_dataset
    )


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "truthlens_logistic_regression.pkl"
)

VECTORIZER_PATH = (
    MODEL_DIR
    / "truthlens_tfidf_vectorizer.pkl"
)

METRICS_PATH = (
    MODEL_DIR
    / "truthlens_metrics.pkl"
)


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():
    """
    Train the TruthLens AI fake-news classifier.

    Machine-learning pipeline:
        Cleaned news text
        -> TF-IDF features
        -> Logistic Regression
        -> Real/Fake prediction

    Labels:
        0 = Real
        1 = Fake
    """

    print("\nLoading dataset...")

    df = load_news_dataset()

    X = df["clean_text"]
    y = df["label"]


    # =====================================================
    # TRAIN / TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # =====================================================
    # TF-IDF FEATURE EXTRACTION
    # =====================================================

    print("Creating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )

    X_test_tfidf = vectorizer.transform(
        X_test
    )


    # =====================================================
    # LOGISTIC REGRESSION MODEL
    # =====================================================

    print(
        "Training Logistic Regression model..."
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train_tfidf,
        y_train
    )


    # =====================================================
    # TEST SET PREDICTIONS
    # =====================================================

    predictions = model.predict(
        X_test_tfidf
    )


    # =====================================================
    # PERFORMANCE METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    # Fake news is label 1, therefore these metrics use
    # the Fake class as the positive class.
    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[
            "Real",
            "Fake"
        ],
        output_dict=True,
        zero_division=0
    )


    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_records": int(len(df)),
        "real_records": int(
            (df["label"] == 0).sum()
        ),
        "fake_records": int(
            (df["label"] == 1).sum()
        ),
        "label_mapping": {
            "0": "Real",
            "1": "Fake"
        },
        "model_name": (
            "TF-IDF + Logistic Regression"
        )
    }


    # =====================================================
    # SAVE TRAINED ARTIFACTS
    # =====================================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    joblib.dump(
        metrics,
        METRICS_PATH
    )


    # =====================================================
    # DISPLAY TRAINING RESULTS
    # =====================================================

    print(
        "\nTruthLens AI Model Training Complete"
    )

    print("-" * 45)

    print(
        f"Training samples : "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples  : "
        f"{len(X_test)}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision        : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall           : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score         : "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(matrix)

    print(
        "\nConfusion matrix layout:"
    )

    print(
        "[[True Real, Real predicted as Fake],"
    )

    print(
        " [Fake predicted as Real, True Fake]]"
    )

    print(
        "\nSaved model:"
    )

    print(MODEL_PATH)

    print(
        "\nSaved vectorizer:"
    )

    print(VECTORIZER_PATH)

    print(
        "\nSaved metrics:"
    )

    print(METRICS_PATH)

    return (
        model,
        vectorizer,
        metrics
    )


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

def load_trained_model():
    """
    Load the saved classifier, TF-IDF vectorizer,
    and evaluation metrics.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained TruthLens AI model was not found. "
            "Run 'python -m src.model' first."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "The TF-IDF vectorizer was not found. "
            "Run 'python -m src.model' first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    metrics = None

    if METRICS_PATH.exists():
        metrics = joblib.load(
            METRICS_PATH
        )

    return (
        model,
        vectorizer,
        metrics
    )


# =========================================================
# PREDICT A SINGLE NEWS ARTICLE
# =========================================================

def predict_news(
    text,
    model,
    vectorizer
):
    """
    Classify supplied news text as Real or Fake.

    The incoming text is cleaned using the exact same
    preprocessing function used during model training.

    Returns:
        dictionary containing:
        - label
        - numeric prediction
        - confidence
        - probability of Real
        - probability of Fake
    """

    if text is None:
        raise ValueError(
            "Please provide news text for analysis."
        )

    cleaned_text = clean_text(
        text
    )

    if len(cleaned_text) < 20:
        raise ValueError(
            "Please provide enough meaningful news text "
            "for analysis."
        )


    # =====================================================
    # TRANSFORM USER TEXT
    # =====================================================

    transformed_text = vectorizer.transform(
        [cleaned_text]
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = int(
        model.predict(
            transformed_text
        )[0]
    )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    probabilities = model.predict_proba(
        transformed_text
    )[0]

    real_probability = float(
        probabilities[0]
    )

    fake_probability = float(
        probabilities[1]
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = float(
        max(
            real_probability,
            fake_probability
        )
    )


    # =====================================================
    # HUMAN-READABLE LABEL
    # =====================================================

    prediction_label = (
        "Fake"
        if prediction == 1
        else "Real"
    )


    # =====================================================
    # RETURN RESULT
    # =====================================================

    return {
        "label": prediction_label,
        "prediction": prediction,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
        "cleaned_text": cleaned_text
    }


# =========================================================
# OPTIONAL MODEL INFORMATION
# =========================================================

def get_model_info():
    """
    Return basic information about the saved model.
    """

    return {
        "model_type": (
            "Logistic Regression"
        ),
        "feature_extraction": (
            "TF-IDF"
        ),
        "labels": {
            0: "Real",
            1: "Fake"
        },
        "model_path": str(
            MODEL_PATH
        ),
        "vectorizer_path": str(
            VECTORIZER_PATH
        )
    }


# =========================================================
# RUN TRAINING
# =========================================================

if __name__ == "__main__":
    train_model()