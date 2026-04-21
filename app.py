import streamlit as st
import pickle
import nltk
nltk.download('stopwords', quiet=True)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# UI
st.title("📰 Fake News Detector")
st.write("Paste any news article below to check if it's **Real or Fake**")

news_input = st.text_area("Enter news article here:", height=250)

if st.button("🔍 Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Predict
        vectorized = vectorizer.transform([news_input])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        if prediction == 1:
            st.success("✅ This news appears to be **REAL**")
        else:
            st.error("🚨 This news appears to be **FAKE**")

        # Confidence bar
        st.write("### Confidence Score")
        col1, col2 = st.columns(2)
        col1.metric("Fake", f"{confidence[0]*100:.1f}%")
        col2.metric("Real", f"{confidence[1]*100:.1f}%")
        