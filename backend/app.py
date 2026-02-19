from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# ----------------------------
# INIT
# ----------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_VERSION = "4.4.0"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
print("Loaded YouTube Key:", YOUTUBE_API_KEY[:10])


# ----------------------------
# LOAD MODEL + ENCODER + TFIDF
# ----------------------------
model = pickle.load(open("model/random_forest_v4.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder_v4.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer_v4.pkl", "rb"))

print("✅ Model + Encoder + TFIDF Loaded")
print("YouTube key loaded:", bool(YOUTUBE_API_KEY))

# ----------------------------
# LOAD DATASET
# ----------------------------
lyrics_df = pd.read_csv("data/music_labeled_v4.csv")

# 🔥 Rename columns properly
lyrics_df = lyrics_df.rename(columns={
    "artist_name": "artist",
    "track_name": "title"
})

lyrics_df = lyrics_df.dropna(subset=["lyrics"]).reset_index(drop=True)

# 🔥 Convert numeric emotion → English if needed
if "emotion" in lyrics_df.columns:
    if lyrics_df["emotion"].dtype != object:
        lyrics_df["emotion"] = label_encoder.inverse_transform(
            lyrics_df["emotion"].astype(int)
        )

print("Loaded dataset:", lyrics_df.shape)

# Precompute TF-IDF vectors once
lyrics_vectors = vectorizer.transform(lyrics_df["lyrics"].astype(str))

# ----------------------------
# CACHE
# ----------------------------
youtube_cache = {}

# ----------------------------
# YOUTUBE ENRICHMENT
# ----------------------------
import re

def clean_text(text):
    text = re.sub(r"\(.*?\)", "", text)  # remove parentheses
    text = re.sub(r"[^a-zA-Z0-9\s-]", "", text)  # remove special chars
    return text.strip()

def enrich_with_youtube(title, artist):
    key = f"{title}-{artist}"

    if key in youtube_cache:
        return youtube_cache[key]

    if not YOUTUBE_API_KEY:
        return {"youtube_url": None, "youtube_embed": None}

    clean_title = clean_text(title)
    clean_artist = clean_text(artist)

    query = f"{clean_artist} - {clean_title}"

    print("🔎 Searching YouTube for:", query)

    try:
        url = "https://www.googleapis.com/youtube/v3/search"

        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 1,
            "type": "video"
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        print("📦 Response:", data)

        if "items" in data and len(data["items"]) > 0:
            video_id = data["items"][0]["id"]["videoId"]

            result = {
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                "youtube_embed": f"https://www.youtube.com/embed/{video_id}"
            }

            youtube_cache[key] = result
            return result

        print("⚠️ No YouTube results found")

    except Exception as e:
        print("🔥 YouTube error:", e)

    fallback = {"youtube_url": None, "youtube_embed": None}
    youtube_cache[key] = fallback
    return fallback

# ----------------------------
# RECOMMENDER ENGINE
# ----------------------------
def recommend_songs(user_text, top_n=5):

    if not user_text:
        return None

    # Vectorize input
    user_vector = vectorizer.transform([str(user_text)])

    # Predict emotion
    predicted_label = model.predict(user_vector)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

    probs = model.predict_proba(user_vector)[0]
    confidence = float(np.max(probs))

    # Similarity
    similarities = cosine_similarity(user_vector, lyrics_vectors)[0]
    similarities = (similarities - similarities.min()) / (
        similarities.max() - similarities.min() + 1e-8
    )

    # Emotion Boost
    emotion_match = (lyrics_df["emotion"] == predicted_emotion).astype(int).values
    final_score = similarities * 0.65 + emotion_match * (0.35 * confidence)

    # Add temperature noise
    noise = np.random.normal(0, 0.05, len(final_score))
    final_score = final_score + noise

    final_score = (final_score - final_score.min()) / (
        final_score.max() - final_score.min() + 1e-8
    )

    temp_df = lyrics_df.copy()
    temp_df["score"] = final_score

    # Candidate pool
    candidate_pool = temp_df.sort_values(by="score", ascending=False).head(60)

    # Artist diversity
    unique_songs = []
    seen_artists = set()

    for _, row in candidate_pool.iterrows():
        artist_name = str(row["artist"])
        if artist_name not in seen_artists:
            unique_songs.append(row)
            seen_artists.add(artist_name)
        if len(unique_songs) == top_n:
            break

    # Enrich
    def enrich(row):
        yt = enrich_with_youtube(str(row["title"]), str(row["artist"]))
        return {
            "title": str(row["title"]),
            "artist": str(row["artist"]),
            "similarity": float(row["score"]),
            "youtube_url": yt["youtube_url"],
            "youtube_embed": yt["youtube_embed"]
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        enriched = list(executor.map(enrich, unique_songs))

    return {
        "model_version": str(MODEL_VERSION),
        "predicted_emotion": str(predicted_emotion),
        "confidence": float(confidence),
        "songs": enriched
    }


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Emotion-Aware Music Recommender API Running",
        "model_version": MODEL_VERSION
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        result = recommend_songs(text)

        if result is None:
            return jsonify({"error": "Invalid input"}), 400

        return jsonify(result)

    except Exception as e:
        print("🔥 BACKEND ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
