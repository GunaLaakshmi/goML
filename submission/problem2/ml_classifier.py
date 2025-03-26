import re
import string
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\s+", " ", text).strip()  # Remove extra spaces
    return text

# Function to train the sentiment model
def train_sentiment_model(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['Review'])
    df['Sentiment'] = df['Recommended'].map({'yes': 'positive', 'no': 'negative'})
    df = df.dropna(subset=['Sentiment'])
    
    texts = df['Review'].apply(preprocess_text).tolist()
    labels = np.array([1 if label == "positive" else 0 for label in df['Sentiment']])
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)  # Convert text to TF-IDF features
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    model = LogisticRegression()
    model.fit(X_train, y_train)  # Train the model
    
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    joblib.dump((model, vectorizer), "sentiment_model.pkl")  # Save model and vectorizer
    return model, vectorizer

# Function to predict sentiment of a new text
def predict_sentiment(model_vectorizer_tuple, new_text):
    model, vectorizer = model_vectorizer_tuple
    new_text = preprocess_text(new_text)
    X_new = vectorizer.transform([new_text])
    prediction = model.predict(X_new)[0]
    return "positive" if prediction == 1 else "negative"

# Dataset Path
dataset_path = r"C:\Users\guna laakshmi\Downloads\submission\problem2\AirlineReviews.csv"


# Train the model using dataset
model_vectorizer_tuple = train_sentiment_model(dataset_path)

# Test Predictions
print(predict_sentiment(model_vectorizer_tuple, "The seats were comfortable and service was great!"))  # Expected: positive
print(predict_sentiment(model_vectorizer_tuple, "They lost my baggage and were very unhelpful!"))  # Expected: negative
