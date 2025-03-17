import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np



# Load the model and TF-IDF vectorizer
model = joblib.load('model.pkl')
tf_idf_vector = joblib.load('tfidf.pkl')

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to predict sentiment
def predict_sentiment(review):
    cleaned_review = re.sub('<.*?>', '', review)
    cleaned_review = re.sub(r'[^\w\s]', '', cleaned_review)
    cleaned_review = cleaned_review.lower()
    tokenized_review = word_tokenize(cleaned_review)
    filtered_review = [word for word in tokenized_review if word not in stop_words]
    stemmed_review = [stemmer.stem(word) for word in filtered_review]
    tfidf_review = tf_idf_vector.transform([' '.join(stemmed_review)])
    sentiment_prediction = model.predict(tfidf_review)
    print(sentiment_prediction)
    predicted_class_index = np.argmax(sentiment_prediction)
    print(predicted_class_index)
    if  predicted_class_index > 0.6:  # Adjust threshold as needed
        return "Positive"
    else:
        return "Negative"

# Streamlit UI
st.title('Chatgpt Review Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", predicted_sentiment)