# Hate Speech Detection Project

A machine learning project to detect and classify hate speech in social media text using NLP techniques and transformer-based models. The model identifies harmful content online and promotes safer digital spaces.

## 🔍 Features

* Detects hate speech, offensive language, and neutral content.
* Trained on **Kaggle datasets** for real-world social media text.
* Implements transformer-based models for accurate classification.
* Provides evaluation metrics and visualizations for performance analysis.

## 🛠️ Technologies & Libraries

* **Python 3.x**
* **PyTorch** / **Transformers (Hugging Face)**
* **scikit-learn** for preprocessing and evaluation
* **Pandas & NumPy** for data handling
* **Matplotlib & Seaborn** for visualization
* **Jupyter Notebook** for experimentation

## ⚙️ Project Structure

```text
HateSpeechDetection/
│
├── data/
│   ├── train.csv          # Kaggle dataset for training
│   ├── test.csv           # Kaggle dataset for testing
│   └── preprocess.py      # Data cleaning & preprocessing
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── model.py           # Model architecture and training
│   ├── train.py           # Training pipeline
│   └── predict.py         # Predict new text
│
├── requirements.txt       # Dependencies
└── README.md
```

## 🏋️‍♂️ Training & Evaluation

* Model trained on Kaggle hate speech dataset.
* Achieved **~93% accuracy** on validation set.
* Evaluates with:

  * Confusion matrix
  * Precision, recall, F1-score

## 📝 Usage

```python
from src.predict import predict_text

text = "Your sample text here"
prediction = predict_text(text)
print(prediction)
```

## 💡 Future Improvements

* Extend to multiple languages.
* Real-time hate speech monitoring for social media.
* Deploy as a web application using Flask or FastAPI.

## 📚 References

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Kaggle Hate Speech Datasets](https://www.kaggle.com/)
* Research papers on hate speech detection and NLP classification


