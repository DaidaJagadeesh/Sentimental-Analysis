# %%
import pandas as pd
import numpy as np

# %%
df = pd.read_csv('IMDB Dataset.csv')

# %%
df

# %%
df['review'].iloc[1]

# %%
import re
df['clean_text'] = df['review'].apply(lambda x:re.sub("<.*?>","",x))

# %%

df['clean_text'].iloc[1]

# %%
df['clean_text'] = df['clean_text'].apply(lambda x:re.sub(r'[^\w\s]', "",x))

# %%
df['clean_text'].iloc[1]

# %%
df['clean_text'] = df['clean_text'].str.lower()

# %%
df['clean_text'].iloc[1]

# %%
# !pip install nltk

# %%
from nltk.tokenize import word_tokenize

# %%
df['tokenize_text'] = df['clean_text'].apply(lambda x:word_tokenize(x))

# %%
df['tokenize_text'].iloc[1]

# %%
import nltk
nltk.download('stopwords')

# %%
from nltk.corpus import stopwords

# %%
stop_words = set(stopwords.words('english'))

# %%
stop_words

# %%
df['filtered_text'] = df['tokenize_text'].apply(lambda x:[word for word in x if word not in stop_words])

# %%
len(df['filtered_text'].iloc[1])

# %%
len(df['tokenize_text'].iloc[1])

# %%
from nltk.stem import PorterStemmer

# %%
stem = PorterStemmer()

# %%
df['stem_text'] = df['filtered_text'].apply(lambda x: [stem.stem(word)for word in x])

# %%
df['stem_text'].iloc[1]

# %%
df['filtered_text'].iloc[1]

# %%
from nltk.stem import WordNetLemmatizer

# %%
lemma = WordNetLemmatizer()

# %%
df['lemma_text'] = df['filtered_text'].apply(lambda x: [lemma.lemmatize(word)for word in x])

# %%
df['lemma_text'].iloc[1]

# %%
X = df['stem_text']
y = df['sentiment']

# %%
y

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_tyrain, y_test = train_test_split(X,y, random_state=42, test_size=0.2)

# %%
X_train

# %%
X_test

# %%
y_tyrain

# %%
y_test

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
tfidf = TfidfVectorizer()

# %%
X_train

# %%
X_train = tfidf.fit_transform(X_train.apply(lambda x:''.join(x)))

# %%
X_test = tfidf.transform(X_test.apply(lambda x: "".join(x)))

# %%
from sklearn.preprocessing import LabelEncoder

# %%
le = LabelEncoder()

y_train = le.fit_transform(y_tyrain)
y_test = le.transform(y_test)

# %%
from keras.utils import to_categorical

# %%
y_train = to_categorical(y_train, num_classes=2)


# %%
y_train

# %%
X_test.shape

# %%
type(X_train)

# %%
from keras import Sequential

# %%
from keras.layers import Dense

# %%
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(2, activation="sigmoid")
])
    

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# %%
model.fit(X_train, y_train, epochs=10)

# %%
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# Assuming your trained model is named 'model'
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')

# Load the model and TF-IDF vectorizer
model = joblib.load('model.pkl')
tf_idf_vector = joblib.load('model.pkl')




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
    if sentiment_prediction > 0.6:  # Adjust threshold as needed
        return "Positive"
        
    else:
        return "Negative"

# Streamlit UI
st.title('Sentiment Analysis')
review_to_predict = st.text_area('Enter your review here:')
if st.button('Predict Sentiment'):
    predicted_sentiment = predict_sentiment(review_to_predict)
    st.write("Predicted Sentiment:", predicted_sentiment)


# %%
!streamlit run NLP.ipynb

# %%
!ipynb-py-convert NLP.ipynb NLP.py

# %%
!pip install ipynb-py-convert

# %%


# %%
