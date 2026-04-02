import streamlit as st
import pickle
import re
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Load saved models (UPDATED ✅)
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ------------------------
# Preprocessing Functions
# ------------------------

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_tags(text):
    return re.sub(r'<.*?>', '', text)

def clean_text(text):
    text = remove_url(text)
    text = remove_tags(text)

    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

    return " ".join(tokens)

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review and get sentiment prediction")

user_input = st.text_area("Enter Review:")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned = clean_text(user_input)

        if cleaned.strip() == "":
            st.error("❌ Input is not valid enter only text")
        else:
            vec = tfidf.transform([cleaned])

            prediction = model.predict(vec)[0]

            # OPTIONAL: Confidence score 🔥
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vec)[0]
                confidence = max(proba)
                st.info(f"Confidence: {confidence:.2f}")

            if prediction == 1:
                st.success("✅ Positive Review 😊")
            else:
                st.error("❌ Negative Review 😡")