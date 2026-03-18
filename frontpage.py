from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# List of hate words for additional check
hate_words = [
    "hole", "nigger", "chink", "spic", "retard", "pedo", "child", "tranny", "faggot", "blacked"
]

# List of offensive words for additional check
offensive_words = [
    "fuck", "shit", "damn", "bitch", "asshole", "bastard", "cunt", "dick", "pussy", "motherfucker",
    "rascal", "idiot", "stupid", "dumb", "moron", "jerk", "prick", "slut", "whore", "faggot",
    "nigger", "chink", "spic", "kike", "wetback", "coon", "gook", "raghead", "sandnigger",
    "nonsense", "boobies", "arsehole", "naked", "gangbang", "rapist", "cock", "twat", "snatch",
    "slit", "cumdump", "cumslut", "fuckhole", "cockslut", "cocksucker", "cumrag", "fucktoy",
    "rape", "breeder", "gutter", "guzzling", "throatfuck", "pukes", "dumpster", "ruin", "little",
    "toy", "gape", "forced", "breeding", "sow", "meat", "bait", "snuff", "scat", "piss", "anal", "rapebait"
]

# List of hate phrases for additional check
hate_phrases = [
    "nigger cunt", "chink whore", "spick slut", "retard fuckpig", "pedo bait", "child cunt",
    "tranny dick", "faggot cocksleeve", "nigger breeder", "blacked whore"
]

# List of offensive phrases for additional check
offensive_phrases = [
    "gangbang whore", "cum guzzling gutter slut", "anal only whore", "piss whore", "scat slut",
    "rape meat", "snuff bait", "destroy her holes", "breed that bitch", "throatfuck until she pukes",
    "turn her into a cum dumpster", "ruin that little rape toy", "gape her asshole", "forced breeding sow"
]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Check for hate phrases
def contains_hate_phrase(text):
    text_lower = text.lower()
    for phrase in hate_phrases:
        if phrase in text_lower:
            return True
    return False

# Check for offensive phrases
def contains_offensive_phrase(text):
    text_lower = text.lower()
    for phrase in offensive_phrases:
        if phrase in text_lower:
            return True
    return False

# Check for hate words
def contains_hate(text):
    text_lower = text.lower()
    for word in hate_words:
        if word in text_lower:
            return True
    return False

# Check for offensive words
def contains_offensive(text):
    text_lower = text.lower()
    for word in offensive_words:
        if word in text_lower:
            return True
    return False

# Load model + vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    tweet = data["tweet"]

    # Check for hate phrases first
    if contains_hate_phrase(tweet):
        pred = 0  # Hate speech
    # Check for offensive phrases second
    elif contains_offensive_phrase(tweet):
        pred = 1  # Offensive language
    # Check for hate words third
    elif contains_hate(tweet):
        pred = 0  # Hate speech
    # Check for offensive words fourth
    elif contains_offensive(tweet):
        pred = 1  # Offensive language
    else:
        cleaned = clean_text(tweet)
        vector = vectorizer.transform([cleaned])
        probs = model.predict_proba(vector)[0]

        # Custom thresholds for better detection
        if probs[0] > 0.4:  # Hate speech threshold
            pred = 0
        elif probs[1] > 0.5:  # Offensive language threshold
            pred = 1
        else:
            pred = 2  # Neutral

    prediction_text = "hate speech" if pred == 0 else "offensive language" if pred == 1 else "neutral"
    return jsonify({
        "class": int(pred),
        "prediction": prediction_text,
        "hate_speech": int(pred == 0),
        "offensive_language": int(pred == 1),
        "neutral": int(pred == 2),
        "tweet": tweet
    })

if __name__ == "__main__":
    app.run(debug=True, port=8000)
