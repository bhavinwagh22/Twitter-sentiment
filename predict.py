import pickle
import re
import numpy as np
import nltk


import nltk
import os

# Create a custom directory for NLTK data to avoid permission issues on the cloud
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Force download the specific missing resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.tokenize import word_tokenize
# ... rest of your imports

# ✅ download FIRST
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ✅ THEN import nltk tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('omw-1.4', quiet=True)

# ---------- LOAD MODEL ----------
model = load_model("sentiment_bilstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 100


# ---------- SAME PREPROCESS AS TRAINING ----------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


# ---------- PREDICTION ----------
def predict_sentiment(text):

    text = preprocess(text)   # ⭐ CRITICAL LINE

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    pred = model.predict(padded, verbose=0)
    label = np.argmax(pred)

    labels = {
        0: "Negative 😡",
        1: "Neutral 😐",
        2: "Positive 😄"
    }

    return labels[label]