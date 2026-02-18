import pandas as pd

df = pd.read_csv("data/train.tsv", sep="\t", header=None)
df.columns = ["text", "labels", "id"]

# Keep only single-label rows for simplicity
df = df[df['labels'].str.contains(",") == False]

# Map label numbers to emotion names
emotion_map = {
    0: "admiration",
    1: "amusement",
    2: "anger",
    3: "annoyance",
    4: "approval",
    5: "caring",
    6: "confusion",
    7: "curiosity",
    8: "desire",
    9: "disappointment",
    10: "disapproval",
    11: "disgust",
    12: "embarrassment",
    13: "excitement",
    14: "fear",
    15: "gratitude",
    16: "grief",
    17: "joy",
    18: "love",
    19: "nervousness",
    20: "optimism",
    21: "pride",
    22: "realization",
    23: "relief",
    24: "remorse",
    25: "sadness",
    26: "surprise"
}

df['emotion'] = df['labels'].astype(int).map(emotion_map)

df = df[['text', 'emotion']]
df.to_csv("data/goemotions_clean.csv", index=False)

print("GoEmotions prepared successfully!")
