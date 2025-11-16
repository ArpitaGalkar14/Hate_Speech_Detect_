import sys, json, joblib, re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

tweet = sys.argv[1]
cleaned = clean_text(tweet)
vector = vectorizer.transform([cleaned])
pred = model.predict(vector)[0]

response = {
    "class": int(pred),
    "hate_speech": int(pred == 0),
    "offensive_language": int(pred == 1),
    "neutral": int(pred == 2),
    "tweet": tweet
}

print(json.dumps(response))

