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

# 2. Preprocessing Function (negation-aware)
def preprocess_text(text):
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")

    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower().split()

    processed_words = []
    i = 0
    while i < len(text):
        word = text[i]
        if word == 'not':
            for j in range(i + 1, min(i + 4, len(text))):
                if text[j] in ['good', 'great', 'bad', 'worth', 'impressive',
                                'engaging', 'excellent', 'amazing', 'terrible', 'awful']:
                    processed_words.append(f'not_{text[j]}')
                    processed_words.append('negation_marker')
                    i = j + 1
                    break
            else:
                processed_words.append(word)
                i += 1
        else:
            if word not in stop_words:
                processed_words.append(lemmatizer.lemmatize(word))
            i += 1

    return ' '.join(processed_words)
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