import streamlit as st
import pandas as pd

from src.model import (
    load_trained_model,
    predict_news
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "analysis_key" not in st.session_state:
    st.session_state.analysis_key = 0


# =========================================================
# RESET
# =========================================================

def start_new_analysis():
    st.session_state.prediction_result = None
    st.session_state.analysis_key += 1


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_resources():
    return load_trained_model()


try:
    model, vectorizer, metrics = load_resources()

except Exception as error:

    st.error(
        "TruthLens AI could not load the trained model."
    )

    st.code(str(error))

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔎 TruthLens AI")

    st.caption(
        "Machine Learning News Classification"
    )

    st.divider()

    st.markdown("### ✨ Features")

    st.write("✓ News text analysis")
    st.write("✓ Real/Fake classification")
    st.write("✓ NLP text preprocessing")
    st.write("✓ TF-IDF feature extraction")
    st.write("✓ Logistic Regression model")
    st.write("✓ Prediction confidence")
    st.write("✓ Class probabilities")
    st.write("✓ Model performance metrics")
    st.write("✓ Start new analysis")

    st.divider()

    st.markdown("### 🧠 ML Pipeline")

    st.caption(
        "News Text → Cleaning → TF-IDF → "
        "Logistic Regression → Prediction"
    )

    st.divider()

    if st.button(
        "🔄 Start New Analysis",
        width="stretch"
    ):
        start_new_analysis()
        st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title("🔎 TruthLens AI")

st.subheader(
    "Machine Learning Fake News Detection"
)

st.write(
    "Analyze news text using a machine-learning model "
    "trained on the ISOT Fake News Dataset."
)

st.warning(
    "TruthLens AI provides a model-based classification, "
    "not a definitive fact-check. Predictions should be "
    "interpreted together with reliable source verification."
)

st.divider()


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("## 📊 Model Performance")

if metrics:

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1-Score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

    st.caption(
        "Metrics are calculated on the held-out ISOT "
        "dataset test split and should not be interpreted "
        "as universal performance on all news sources."
    )

    with st.expander(
        "View evaluation details"
    ):

        st.write(
            f"**Training samples:** "
            f"{metrics['train_size']:,}"
        )

        st.write(
            f"**Testing samples:** "
            f"{metrics['test_size']:,}"
        )

        matrix = metrics[
            "confusion_matrix"
        ]

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual Real",
                "Actual Fake"
            ],
            columns=[
                "Predicted Real",
                "Predicted Fake"
            ]
        )

        st.markdown(
            "#### Confusion Matrix"
        )

        st.dataframe(
            matrix_df,
            width="stretch"
        )


st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("## 📰 Analyze News")

st.write(
    "Enter a news headline and article text below."
)

title = st.text_input(
    "News headline",
    placeholder=(
        "Example: Government announces new policy..."
    ),
    key=f"title_{st.session_state.analysis_key}"
)

article = st.text_area(
    "News article text",
    placeholder=(
        "Paste the article content here. Longer and more "
        "complete text generally provides more useful input "
        "for the classifier."
    ),
    height=250,
    key=f"article_{st.session_state.analysis_key}"
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze News",
    type="primary",
    width="stretch"
):

    combined_text = (
        f"{title.strip()} "
        f"{article.strip()}"
    ).strip()

    if len(combined_text) < 40:

        st.warning(
            "Please enter a meaningful headline or a longer "
            "article before running the analysis."
        )

    else:

        try:

            with st.spinner(
                "TruthLens AI is analyzing the news text..."
            ):

                result = predict_news(
                    combined_text,
                    model,
                    vectorizer
                )

            st.session_state.prediction_result = result

        except Exception as error:

            st.error(
                "The news text could not be analyzed."
            )

            st.code(str(error))


# =========================================================
# RESULTS
# =========================================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()

    st.markdown("## 🎯 Analysis Result")

    label = result["label"]
    confidence = result["confidence"]

    if label == "Fake":

        st.error(
            "⚠️ Model Classification: FAKE"
        )

        st.write(
            "The model found the supplied text more similar "
            "to patterns learned from the fake-news class "
            "in its training dataset."
        )

    else:

        st.success(
            "✅ Model Classification: REAL"
        )

        st.write(
            "The model found the supplied text more similar "
            "to patterns learned from the real-news class "
            "in its training dataset."
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            label
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.1f}%"
        )

    with col3:

        higher_probability = max(
            result["real_probability"],
            result["fake_probability"]
        )

        st.metric(
            "Highest Probability",
            f"{higher_probability * 100:.1f}%"
        )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.markdown(
        "### 📈 Class Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Class": [
                "Real",
                "Fake"
            ],
            "Probability": [
                result["real_probability"],
                result["fake_probability"]
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index(
            "Class"
        )
    )

    probability_display = pd.DataFrame(
        {
            "Class": [
                "Real",
                "Fake"
            ],
            "Probability": [
                f"{result['real_probability'] * 100:.2f}%",
                f"{result['fake_probability'] * 100:.2f}%"
            ]
        }
    )

    st.dataframe(
        probability_display,
        hide_index=True,
        width="stretch"
    )


    # =====================================================
    # RESPONSIBLE INTERPRETATION
    # =====================================================

    st.markdown(
        "### ℹ️ How to Interpret This Result"
    )

    st.info(
        "This result is generated from linguistic patterns "
        "learned from the training dataset. The model does "
        "not independently verify events, sources, dates, "
        "quotes, or factual claims. Important news should "
        "always be checked against reliable independent sources."
    )


    # =====================================================
    # NEW ANALYSIS
    # =====================================================

    st.divider()

    if st.button(
        "🔄 Analyze Another Article",
        width="stretch"
    ):

        start_new_analysis()
        st.rerun()


# =========================================================
# ABOUT
# =========================================================

st.divider()

with st.expander(
    "ℹ️ About TruthLens AI"
):

    st.write(
        """
        TruthLens AI uses Natural Language Processing and
        supervised machine learning to classify news text.

        **Model pipeline:**

        1. Text cleaning and normalization  
        2. TF-IDF feature extraction  
        3. Logistic Regression classification  
        4. Real/Fake probability estimation  

        **Labels used during training:**

        - `0 = Real`
        - `1 = Fake`

        The model was trained using the ISOT Fake News Dataset.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "TruthLens AI • NLP + TF-IDF + Logistic Regression • "
    "Developed for the SAM AI Technologies AI Internship"
)