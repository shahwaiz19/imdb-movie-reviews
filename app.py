# streamlit_sentiment_app.py
# Streamlit web app for IMDB sentiment classification
# Requirements: streamlit, scikit-learn, pandas, numpy

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download('stopwords')
nltk.download('punkt')


st.set_page_config(page_title="IMDB Sentiment Classifier", layout="wide")

# ---------- Helper functions ----------
@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_and_vectorizer(model_path="sentiment_model.pkl", vec_path="tfidf_vectorizer.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None, f"Missing files: {model_path if not os.path.exists(model_path) else ''} {vec_path if not os.path.exists(vec_path) else ''}".strip()
    model = load_pickle(model_path)
    tfidf = load_pickle(vec_path)
    return model, tfidf, None


def predict_text(model, tfidf, text):
    vect = tfidf.transform([text])
    pred = model.predict(vect)[0]
    try:
        prob = model.predict_proba(vect)[0]
        # prob[1] is probability for positive class (assuming mapping 1=positive)
        pos_prob = float(prob[1])
    except Exception:
        pos_prob = None
    return pred, pos_prob


def get_top_words_for_class(model, tfidf, class_label=1, top_n=15):
    # For MultinomialNB, feature_log_prob_ gives log prob of features per class
    if not hasattr(model, "feature_log_prob_"):
        return []
    feature_names = tfidf.get_feature_names_out()
    # class_label should be 0 or 1
    log_probs = model.feature_log_prob_[class_label]
    top_idx = np.argsort(log_probs)[-top_n:][::-1]
    return [(feature_names[i], float(log_probs[i])) for i in top_idx]


# ---------- UI ----------
st.title("🎬 IMDB Movie Review Sentiment - Streamlit App")
st.markdown(
    "This app uses a trained **Multinomial Naive Bayes** model (TF-IDF features) to predict whether an IMDB review is **positive** or **negative**.\n\n"
    "Place `sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as this app, or upload custom files via the sidebar."
)

# Sidebar: Model loading
st.sidebar.header("Model & Vectorizer")
use_default = st.sidebar.checkbox("Use local files in project folder (sentiment_model.pkl, tfidf_vectorizer.pkl)", value=True)

uploaded_model = st.sidebar.file_uploader("Upload model (pickle)", type=["pkl"])
uploaded_vec = st.sidebar.file_uploader("Upload TF-IDF vectorizer (pickle)", type=["pkl"])

model = None
tfidf = None
load_error = None

if uploaded_model is not None and uploaded_vec is not None:
    try:
        model = pickle.load(uploaded_model)
        tfidf = pickle.load(uploaded_vec)
    except Exception as e:
        load_error = f"Failed to load uploaded files: {e}"
elif use_default:
    model, tfidf, err = load_model_and_vectorizer()
    if err:
        load_error = err
else:
    load_error = "No model/vectorizer provided. Choose files or enable local files." 

if load_error:
    st.sidebar.error(load_error)
else:
    st.sidebar.success("Model & vectorizer ready")

# Sidebar: Options
st.sidebar.markdown("---")
show_top = st.sidebar.checkbox("Show top words for each class", value=True)
num_top = st.sidebar.slider("Top words", min_value=5, max_value=40, value=15)

# Main panel: Text input
st.subheader("Write or paste a movie review")
user_text = st.text_area("Enter review text:", height=160)

col1, col2 = st.columns([2,1])
with col1:
    if st.button("Predict"):
        if model is None or tfidf is None:
            st.error("Model or TF-IDF vectorizer not loaded. Provide files in sidebar or place them in the app folder.")
        elif not user_text or user_text.strip()=="":
            st.warning("Please enter a review to predict.")
        else:
            # Preprocessing: minimal cleaning mirroring training (lowercase & remove non-alpha)
            import re
            clean = user_text.lower()
            clean = re.sub(r"[^a-zA-Z\s]", "", clean)
            pred, pos_prob = predict_text(model, tfidf, clean)
            label = "Positive" if pred==1 else "Negative"
            st.markdown(f"### Prediction: **{label}**")
            if pos_prob is not None:
                st.metric("Positive probability", f"{pos_prob:.4f}")

            # Show top words for predicted class
            if show_top:
                with st.expander("Top words for predicted class"):
                    top = get_top_words_for_class(model, tfidf, class_label=int(pred), top_n=num_top)
                    if top:
                        df_top = pd.DataFrame(top, columns=["word", "log_prob"]).set_index("word")
                        st.write(df_top)
                        st.bar_chart(df_top["log_prob"])
                    else:
                        st.write("Top words not available for this model")

with col2:
    st.markdown("## Quick examples")
    examples = [
        "This movie was absolutely fantastic with great acting and a moving story.",
        "Waste of time. The plot was boring and the acting was terrible.",
        "Loved every minute—cinematography and soundtrack were superb!",
        "I couldn't finish it. So dull and predictable."
    ]
    for ex in examples:
        if st.button(f"Use: {ex[:30]}..."):
            user_text = st.session_state.get("user_text", "")
            # replace text area content by setting st.experimental_set_query_params? Instead, use js-free: re-render
            # Streamlit doesn't currently provide a direct way to set text_area value without session state key.
            st.experimental_set_query_params(_text=ex)
            st.write("Copied example into browser query params — paste it into the text box and press Predict.")

# Footer: About & run instructions
st.markdown("---")
st.info("How to run: `pip install streamlit scikit-learn pandas numpy` then `streamlit run streamlit_sentiment_app.py`")
st.caption("If you saved your model/vectorizer with different filenames, either rename them to `sentiment_model.pkl` and `tfidf_vectorizer.pkl` or upload them via the sidebar.")
