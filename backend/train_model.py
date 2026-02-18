import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import extract_features

os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/goemotions_train_ready.csv")

# ---- EMOTION CLUSTER MAPPING ----
emotion_cluster = {
    17: "joy",
    13: "joy",
    1: "joy",

    25: "sadness",
    16: "sadness",

    2: "anger",
    3: "anger",

    18: "love",
    5: "love",

    14: "fear",
    19: "fear",

    27: "neutral"
}

df = df[df["label"].isin(emotion_cluster.keys())]
df["emotion"] = df["label"].map(emotion_cluster)

print("After clustering:", df["emotion"].value_counts())

# Feature extraction
df["features"] = df["text"].apply(extract_features)
df = df.dropna()

X = list(df["features"])
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

with open("model/random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")
