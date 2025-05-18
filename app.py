import streamlit as st
import pickle

# Load trained pipeline
with open("model1.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Sentiment Analyzer", page_icon="📝", layout="centered")
st.title("📝 Sentiment Analyzer")
st.markdown("Enter a sentence to classify it as **Positive**, **Negative**, or **Neutral**.")

# Input
user_input = st.text_area("Write your review below:")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please enter something.")
    else:
        prediction = model.predict([user_input])[0]
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = label_map.get(prediction, "Unknown")
        st.success(f"**Predicted Sentiment:** {sentiment}")
