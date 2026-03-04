import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Page Setup
st.set_page_config(page_title="Sentiment AI")
st.title("Sentiment Analysis App")
st.write("Enter a movie review to predict its sentiment.")

# 2. Preprocessing Function
def preprocess_text(text):
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass
        
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    if "not" in stop_words:
        stop_words.remove("not")
    
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)
user_review = st.text_area("Review Content:")

if st.button("Predict"):
    if user_review.strip():
        cleaned_text = preprocess_text(user_review)
        vectorized_data = cv.transform([cleaned_text]).toarray()
        prediction = model.predict(vectorized_data)[0]
        
        if prediction == 1:
            st.success("Result: Positive Sentiment ✨")
        else:
            st.error("Result: Negative Sentiment 🚫")
    else:
        st.warning("Please enter some text.")
