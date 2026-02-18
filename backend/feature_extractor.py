import nltk
import numpy as np
import re
from collections import Counter
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

nltk.download('punkt')

def preprocess(text):
    text = str(text).lower()
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return sentences, words
def extract_features(text):
    sentences, words = preprocess(text)

    if len(sentences) == 0 or len(words) == 0:
        return None

    avg_sentence_length = np.mean([len(s.split()) for s in sentences])
    punctuation_count = len(re.findall(r'[!?]', text))
    ellipsis_count = text.count("...")
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    lexical_diversity = len(set(words)) / len(words)
    repetition_score = max(Counter(words).values())
    readability = textstat.flesch_reading_ease(text)

    # NEW: Sentiment polarity
    sentiment = analyzer.polarity_scores(text)["compound"]

    return [
        avg_sentence_length,
        punctuation_count,
        ellipsis_count,
        caps_ratio,
        lexical_diversity,
        repetition_score,
        readability,
        sentiment
    ]
