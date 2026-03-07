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
    "hole", "nigger", "chink", "spic", "retard", "pedo", "child", "tranny", "faggot", "blacked",
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
    "tranny dick", "faggot cocksleeve", "nigger breeder", "blacked whore","lust",
    "[group] should not exist.",
    "I hate [group].",
    "All [group] are terrible.",
    "[group] are ruining everything.",
    "Nobody wants [group] here.",
    "[group] are a problem.",
    "Keep [group] away from us.",
    "[group] don't belong here.",
    "I can't stand [group].",
    "[group] are the worst.",
    "[group] are destroying society.",
    "We don't need [group].",
    "[group] are inferior.",
    "[group] are disgusting.",
    "[group] are lazy.",
    "[group] are dangerous.",
    "[group] are not welcome.",
    "Ban all [group].",
    "[group] should go back.",
    "[group] are ruining this country.",
    "I will never accept [group].",
    "[group] don't deserve respect.",
    "[group] are always wrong.",
    "I don't trust [group].",
    "[group] are a mistake.",
    "[group] are useless.",
    "[group] are parasites.",
    "[group] cause all problems.",
    "[group] should leave immediately.",
    "[group] are unwanted.",
    "[group] are harmful.",
    "[group] are nothing but trouble.",
    "[group] don't deserve rights.",
    "[group] are a burden.",
    "[group] are evil.",
    "[group] make everything worse.",
    "[group] are not human.",
    "[group] should be removed.",
    "[group] are bad people.",
    "Stay away from [group].",
    "[group] are toxic.",
    "[group] are a threat.",
    "[group] are ruining our future.",
    "[group] should be banned.",
    "[group] are inferior to us.",
    "[group] are not equal.",
    "[group] are the problem.",
    "[group] should not be trusted.",
    "[group] are destroying culture.",
    "[group] are unwanted in society."

]

# List of offensive phrases for additional check
offensive_phrases = [
    "gangbang whore", "cum guzzling gutter slut", "anal only whore", "piss whore", "scat slut",
    "rape meat", "snuff bait", "destroy her holes", "breed that bitch", "throatfuck until she pukes",
    "turn her into a cum dumpster", "ruin that little rape toy", "gape her asshole", "forced breeding sow","failure"
]

harassment_words = [
    "worthless",
    "disgusting",
    "useless",
    "pathetic",
    "loser",
    "You are useless.",
    "Nobody cares about you.",
    "You are pathetic.",
    "That was stupid.",
    "You are embarrassing.",
    "Stop acting dumb.",
    "You are a failure.",
    "This is the worst thing ever.",
    "You have no idea what you're doing.",
    "That was completely ridiculous.",
    "You always mess things up.",
    "You are annoying.",
    "Nobody likes you.",
    "You are so incompetent.",
    "That was a terrible attempt.",
    "You are hopeless.",
    "You are unbelievably bad at this.",
    "This is trash.",
    "You are not smart at all.",
    "You should just quit.",
    "You make everything worse.",
    "That was a dumb decision.",
    "You are extremely irritating.",
    "You are not worth the effort.",
    "You are disappointing.",
    "You ruined everything.",
    "This is absolute nonsense.",
    "You are clueless.",
    "You don't deserve any credit.",
    "That was awful.",
    "You are weak.",
    "You are so lazy.",
    "You are a joke.",
    "Nobody respects you.",
    "You are totally incompetent.",
    "You can't do anything right.",
    "You are unbearable.",
    "This is embarrassing.",
    "You lack basic skills.",
    "You are terrible at this.",
    "You are frustrating.",
    "You are completely wrong.",
    "You are ignorant.",
    "That idea is horrible.",
    "You are the worst.",
    "You are not good enough.",
    "You are so bad at this.",
    "You are ridiculous.",
    "You should be ashamed.",
    "You are wasting everyone's time."

]

def normalize_text(text):
    text = text.lower()

    # remove common obfuscation characters COMPLETELY
    text = re.sub(r'[@._\-*]+', '', text)

    # replace number-letter swaps
    replacements = {
        '4': 'a',
        '3': 'e',
        '1': 'i',
        '!': 'i',
        '0': 'o',
        '$': 's',
        '5': 's'
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text





def clean_text(text):
    text = str(text).lower()

    # 1️⃣ Remove everything that is NOT a letter
    # This removes @ $ 3 4 5 0 . * - _ etc
    text = re.sub(r'[^a-z\s]', '', text)


    # 2️⃣ Reduce repeated letters (fuuuuck → fuck)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
     # 🔥 REMOVE ALL SPACES
    text = text.replace(" ", "")


    return text

def preprocess_for_matching(text):
    text = normalize_text(text)
    text = clean_text(text)
    text = re.sub(r"\s+", "", text)
   # remove ALL spaces
    return text
def contains_harassment(text):
    processed = preprocess_for_matching(text)
    for word in harassment_words:
        if word in processed:
            return True
    return False


# Check for hate phrases
def contains_hate_phrase(text):
    processed = preprocess_for_matching(text)
    for phrase in hate_phrases:
        phrase_clean = phrase.replace(" ", "")
        if phrase_clean in processed:
            return True
    return False

# Check for offensive phrases
def contains_offensive_phrase(text):
    processed = preprocess_for_matching(text)
    for phrase in offensive_phrases:
        phrase_clean = phrase.replace(" ", "")
        if phrase_clean in processed:
            return True
    return False


# Check for hate words
def contains_hate(text):
    processed = preprocess_for_matching(text)
    for word in hate_words:
        if word in processed:
            return True
    return False


# Check for offensive words
def contains_offensive(text):
    processed = preprocess_for_matching(text)
    for word in offensive_words:
        if word in processed:
            return True
    return False


# Load model + vectorizer
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Hate Speech Detection API. Send POST requests to /predict with JSON: {'tweet': 'your text'}"

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
    elif contains_harassment(tweet):
      pred = 1  # treat as offensive language

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
