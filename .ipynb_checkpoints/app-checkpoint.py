import streamlit as st
import pickle

st.title("🎬 Movie Review Sentiment Analyzer")

# Text box
review = st.text_area("Enter your movie review:")

# Load model and vectorizer once
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vect = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vect

model, vectorizer = load_artifacts()

# Button
if st.button("Analyze"):
    if review.strip():
        # Transform input
        X = vectorizer.transform([review])
        pred = model.predict(X)[0]

        # Handle both number labels and string labels
        if isinstance(pred, str):
            label = pred.lower()
        else:
            label = "positive" if pred == 1 else "negative"

        # Show result
        if label == "positive":
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")

        # Debug: show raw prediction and probability (optional)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write(f"🔎 Probabilities → Negative: {proba[0]:.2f}, Positive: {proba[1]:.2f}")
        st.write("Raw prediction:", pred)

    else:
        st.warning("⚠️ Please type a review first")
