import sys, json, joblib, re
import nltk
from nltk.corpus import stopwords
from better_profanity import profanity  



nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

import unicodedata

def normalize_obfuscated_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)

    substitutions = {
        '@': 'a',
        '4': 'a',
        '3': 'e',
        '1': 'i',
        '!': 'i',
        '0': 'o',
        '$': 's',
        '5': 's',
        '*': '',
    }

    for k, v in substitutions.items():
        text = text.replace(k, v)

    # f.u.c.k → fuck
    text = re.sub(r'(?<=\w)[^a-z\s](?=\w)', '', text)

    # remove spaces between single letters (f u c k → fuck)
    text = re.sub(r'\b(?:[a-z]\s+){2,}[a-z]\b', lambda m: m.group(0).replace(" ", ""), text)


    # fuuuuuck → fuck
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 🔥 REMOVE ALL SPACES
    text = text.replace(" ", "")

    return text


def clean_text(text):
    text = str(text).lower()

    # 🔴 normalize obfuscated profanity
    text = re.sub(r'[@$*!]', 'a', text)
    text = re.sub(r'0', 'o', text)
    text = re.sub(r'1', 'i', text)
    text = re.sub(r'3', 'e', text)
    text = re.sub(r'5', 's', text)
    text = re.sub(r'7', 't', text)

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = " ".join([word for word in text.split() if word not in stop_words])
    return text


model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


tweet = sys.argv[1]

# Normalize first
normalized = normalize_obfuscated_text(tweet)

# Clean for ML model
cleaned = clean_text(normalized)

# 🔥 HARD PROFANITY OVERRIDE
if profanity.contains_profanity(normalized):
    pred = 1  # offensive language
else:
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


