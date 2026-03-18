import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nltk.download('stopwords')
df = pd.read_csv("train.csv")
df.head()
print(df.info())
print(df['class'].value_counts())
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)      # remove URLs
    text = re.sub(r'@\w+', '', text)               # remove @mentions
    text = re.sub(r'[^a-zA-Z\s]', '', text)        # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()       # remove extra spaces

    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df['clean_tweet'] = df['tweet'].apply(clean_text)
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_tweet'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

def predict_hate(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = nb_model.predict(vectorized)[0]
    
    labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
    return labels[pred]

print(predict_hate("I hate you"))
print(predict_hate("Have a nice day"))





