import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset
@st.cache
def load_data():
    data = pd.read_csv("Corona_NLP_train.csv", encoding="latin-1")
    data = data.drop(columns=["UserName", "ScreenName", "Location", "TweetAt"], errors="ignore")
    data = data.dropna()
    return data

# Preprocess Data
def preprocess_data(data):
    data = data.copy()
    data["Sentiment"] = data["Sentiment"].map({
        "Extremely Negative": "Negative",
        "Extremely Positive": "Positive"
    }).fillna(data["Sentiment"])
    return data

# Train Model
@st.cache
def train_model(data):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data["OriginalTweet"])
    y = data["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

# Streamlit UI
st.title("Tweet Sentiment Analysis")
st.write("Analyze the sentiment of tweets during the COVID-19 pandemic.")

# Load and preprocess the data
data = load_data()
data = preprocess_data(data)

if st.checkbox("Show Raw Data"):
    st.write(data.head())

# Train model
model, vectorizer, accuracy = train_model(data)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Input for prediction
user_input = st.text_area("Enter a tweet for sentiment analysis:")
if st.button("Predict"):
    if user_input:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        st.write(f"The sentiment of the tweet is: **{prediction}**")
    else:
        st.write("Please enter a tweet to analyze.")




