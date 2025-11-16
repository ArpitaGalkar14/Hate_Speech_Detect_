from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load model + vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tweet = data["tweet"]

    cleaned = clean_text(tweet)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    return jsonify({
        "class": int(pred),
        "hate_speech": int(pred == 0),
        "offensive_language": int(pred == 1),
        "neutral": int(pred == 2),
        "tweet": tweet
    })

if __name__ == "__main__":
    app.run(debug=True)
