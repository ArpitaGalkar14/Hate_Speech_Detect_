from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
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

class Input(BaseModel):
    tweet: str

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(data: Input):
    cleaned = clean_text(data.tweet)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    return {
        "class": int(pred),
        "hate_speech": int(pred == 0),
        "offensive_language": int(pred == 1),
        "neutral": int(pred == 2),
        "tweet": data.tweet
    }
