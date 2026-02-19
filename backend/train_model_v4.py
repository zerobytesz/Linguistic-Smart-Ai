# train_model_v4.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading GoEmotions dataset...")

df = pd.read_csv("data/goemotions_train_ready.csv")

# Rename for clarity
df = df.rename(columns={"label": "emotion"})

# Clean
df = df.dropna().reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Unique emotions:", df["emotion"].nunique())

# ----------------------------
# TF-IDF VECTORIZATION
# ----------------------------
print("Vectorizing text with TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2
)

X = vectorizer.fit_transform(df["text"].astype(str))
y = df["emotion"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ----------------------------
# TRAIN RANDOM FOREST
# ----------------------------
print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# EVALUATION
# ----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# SAVE MODEL + ENCODER + VECTORIZER
# ----------------------------
pickle.dump(model, open("model/random_forest_v4.pkl", "wb"))
pickle.dump(label_encoder, open("model/label_encoder_v4.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer_v4.pkl", "wb"))

print("\n✅ Saved random_forest_v4.pkl")
print("✅ Saved label_encoder_v4.pkl")
print("✅ Saved tfidf_vectorizer_v4.pkl")
