import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features

# Load trained model
model = pickle.load(open("model/random_forest.pkl", "rb"))

# Load lyrics dataset
lyrics_df = pd.read_csv("data/lyrics_final_clean.csv")

# Precompute lyric features (only once)
print("Extracting features for lyrics...")

lyrics_df["features"] = lyrics_df["lyrics"].apply(extract_features)
lyrics_df = lyrics_df.dropna()

print("Lyrics ready:", len(lyrics_df))


def recommend_songs(user_text, top_n=5):
    user_features = extract_features(user_text)

    if user_features is None:
        return []

    # Predict emotion
    predicted_emotion = model.predict([user_features])[0]

    print("Predicted Emotion:", predicted_emotion)

    # Compute similarity
    lyric_features_matrix = np.vstack(lyrics_df["features"].values)
    user_vector = np.array(user_features).reshape(1, -1)

    similarities = cosine_similarity(user_vector, lyric_features_matrix)[0]

    lyrics_df["similarity"] = similarities

    # Get top songs
    top_songs = lyrics_df.sort_values(
        by="similarity",
        ascending=False
    ).head(top_n)

    return {
        "predicted_emotion": predicted_emotion,
        "songs": top_songs[["title", "artist", "similarity"]].to_dict(
            orient="records"
        )
    }
