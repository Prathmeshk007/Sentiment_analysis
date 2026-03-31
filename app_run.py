import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import sklearn

# Download NLTK resources once at the top
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title="Sentiment Analysis", page_icon="", layout="centered")
st.title(" Sentiment Analysis App for movies")

# Force complete cache clearing
st.cache_data.clear()
st.cache_resource.clear()

# Debug environment info
st.write("Library versions:", {
    "streamlit": st.__version__,
    "sklearn": sklearn.__version__,
    "nltk": nltk.__version__,
})

# Force model reload without caching
def load_model_fresh():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load model fresh every time
model, vectorizer = load_model_fresh()
st.success("✅ Model loaded successfully")

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")   # keep "not" for negation handling

    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower().split()
    processed_words = []
    i = 0
    while i < len(text):
        word = text[i]
        if word == 'not':
            for j in range(i + 1, min(i + 6, len(text))):
                if text[j] in [
                    'good','great','bad','worth','impressive','engaging','excellent','amazing',
                    'terrible','awful','lengthy','boring','poor','disappointing','worse','hate',
                    'annoying','frustrating','ugly','disgusting','horrible','pathetic','stupid',
                    'dumb','lame','weak','trash','garbage'
                ]:
                    processed_words.append(f'not_{text[j]}')
                    processed_words.append('negation_marker')
                    i = j + 1
                    break
            else:
                processed_words.append(word)
                i += 1
        else:
            if word not in stop_words and len(word) > 1:
                processed_words.append(lemmatizer.lemmatize(word))
            i += 1
    
    return ' '.join(processed_words)

# Quick test cases
test_cases = ["actor is ugly", "movie so not good", "movie is good", "the acting was not bad"]
selected = st.selectbox("Quick test:", test_cases)

if st.button("Test Selected"):
    processed = preprocess_text(selected)
    st.write("Processed text:", processed)  # ✅ Debug output

    vectorized = vectorizer.transform([processed]).toarray()
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.write(f"**{selected}** → {sentiment}")
    
    # Debug info
    confidence = max(model.predict_proba(vectorized)[0]) * 100
    st.write(f"🔧 Confidence: {confidence:.1f}%")

# User input
user_input = st.text_area("Enter your review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        st.write("Processed text:", processed)  # ✅ Debug output

        vectorized = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vectorized)[0]
        confidence = max(model.predict_proba(vectorized)[0]) * 100
        
        if prediction == 1:
            st.success(f"✨ **Positive Sentiment** - Confidence: {confidence:.1f}%")
        else:
            st.error(f"🚫 **Negative Sentiment** - Confidence: {confidence:.1f}%")
        st.balloons()