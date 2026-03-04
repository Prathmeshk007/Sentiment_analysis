link=
# Sentiment Analysis Movie Reviews 🎭

A simple machine learning project that predicts the sentiment of movie reviews (Positive or Negative) using a Logistic Regression model and a Streamlit web interface.

## 🚀 Features
- **Machine Learning Model**: Logistic Regression trained on movie review data.
- **Preprocessing**: Custom NLP cleaning (lowercasing, punctuation removal, lemmatization, and stopword filtering).
- **Web Interface**: Clean, easy-to-use UI built with Streamlit.
- **Real-time Prediction**: Instant results for any review text.

## 📁 Project Structure
- `app.py`: The main Streamlit application script.
- `model.pkl`: The trained Logistic Regression model.
- `vectorizer.pkl`: The CountVectorizer used to convert text to numeric data.
- `finalreview.csv`: The dataset used for training/updating the model.
- `sentiment_analysis.ipynb`: The original Jupyter Notebook with data exploration and training.

## 🛠️ Installation

1. **Clone or download** this project folder to your local machine.
2. **Install the required libraries**:
   ```bash
   pip install streamlit pandas scikit-learn nltk
   ```

## 🏃 How to Run

1. Open your terminal or Command Prompt.
2. Navigate to the project directory.
3. Run the following command:
   ```bash
   python -m streamlit run app.py
   ```
4. The app will open in your default web browser.

## 📝 Usage
- Type or paste a movie review into the text area.
- Click the **Predict** button.
- The app will display whether the sentiment is **Positive ✨** or **Negative 🚫**.

## 💡 Note on Accuracy
This model is trained on a small, curated dataset. For broader use cases, it can be retrained on larger datasets like the IMDB dataset (50,000 reviews).
