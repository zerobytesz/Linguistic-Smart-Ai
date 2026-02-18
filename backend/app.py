from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import extract_features
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

MODEL_VERSION = "1.5.0"
# ----------------------------
# MANUAL CORS HANDLING
# ----------------------------

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_response("")
        response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response

# 🔑 PUT YOUR YOUTUBE API KEY HERE

# ----------------------------
# LOAD MODEL
# ----------------------------
model = pickle.load(open("model/random_forest.pkl", "rb"))

# ----------------------------
# LOAD DATASET
# ----------------------------
lyrics_df = pd.read_csv("data/lyrics_final_clean.csv")
print("Loaded lyrics dataset:", lyrics_df.shape)

# ----------------------------
# SENTIMENT ANALYZER
# ----------------------------
analyzer = SentimentIntensityAnalyzer()

print("Extracting features for all lyrics...")
lyrics_df["features"] = lyrics_df["lyrics"].apply(extract_features)
lyrics_df = lyrics_df.dropna().reset_index(drop=True)

lyrics_df["sentiment"] = lyrics_df["lyrics"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)

def map_song_emotion(score):
    if score > 0.3:
        return "joy"
    elif score < -0.3:
        return "sadness"
    else:
        return "neutral"

lyrics_df["emotion"] = lyrics_df["sentiment"].apply(map_song_emotion)

print("Lyrics ready:", lyrics_df.shape)

# ----------------------------
# CACHES
# ----------------------------
deezer_cache = {}
youtube_cache = {}

# ----------------------------
# DEEZER ENRICHMENT
# ----------------------------
def enrich_with_deezer(title, artist):
    key = f"{title}-{artist}"

    if key in deezer_cache:
        return deezer_cache[key]

    try:
        url = "https://api.deezer.com/search"
        query = f'track:"{title}" artist:"{artist}"'
        response = requests.get(url, params={"q": query}, timeout=5)
        data = response.json()

        if data.get("data"):
            track = data["data"][0]
            result = {
                "deezer_url": track.get("link"),
                "preview_url": track.get("preview"),
                "album_image": track.get("album", {}).get("cover_medium")
            }
            deezer_cache[key] = result
            return result

    except:
        pass

    fallback = {
        "deezer_url": None,
        "preview_url": None,
        "album_image": None
    }

    deezer_cache[key] = fallback
    return fallback

# ----------------------------
# YOUTUBE ENRICHMENT
# ----------------------------
def enrich_with_youtube(title, artist):
    key = f"{title}-{artist}"

    if key in youtube_cache:
        return youtube_cache[key]

    try:
        url = "https://www.googleapis.com/youtube/v3/search"

        # More flexible query
        query = f"{artist} {title} official music"

        params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 3,
            "type": "video"
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            for item in data["items"]:
                video_id = item["id"]["videoId"]

                # Basic filtering to avoid shorts
                if len(video_id) == 11:
                    result = {
                        "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                        "youtube_embed": f"https://www.youtube.com/embed/{video_id}"
                    }
                    youtube_cache[key] = result
                    return result

    except Exception as e:
        print("YouTube error:", e)

    fallback = {
        "youtube_url": None,
        "youtube_embed": None
    }

    youtube_cache[key] = fallback
    return fallback



# ----------------------------
# RECOMMENDATION FUNCTION
# ----------------------------
def recommend_songs(user_text, top_n=5):

    user_features = extract_features(user_text)
    if user_features is None:
        return None

    predicted_emotion = model.predict([user_features])[0]
    user_vector = np.array(user_features).reshape(1, -1)

    filtered_df = lyrics_df

    if predicted_emotion in ["joy", "sadness"]:
        filtered_df = lyrics_df[lyrics_df["emotion"] == predicted_emotion]
        if len(filtered_df) < 10:
            filtered_df = lyrics_df

    lyric_features_matrix = np.vstack(filtered_df["features"].values)
    similarities = cosine_similarity(user_vector, lyric_features_matrix)[0]

    similarities = (similarities - similarities.min()) / (
        similarities.max() - similarities.min() + 1e-8
    )

    if predicted_emotion == "joy":
        similarities += (filtered_df["sentiment"].values * 0.3)
    elif predicted_emotion == "sadness":
        similarities -= (filtered_df["sentiment"].values * 0.3)

    similarities = (similarities - similarities.min()) / (
        similarities.max() - similarities.min() + 1e-8
    )

    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarities

    sorted_df = filtered_df.sort_values(by="similarity", ascending=False)

    unique_songs = []
    seen_artists = set()

    for _, row in sorted_df.iterrows():
        if row["artist"] not in seen_artists:
            unique_songs.append(row)
            seen_artists.add(row["artist"])
        if len(unique_songs) == top_n:
            break

    # 🔥 PARALLEL ENRICHMENT
    def enrich_row(row):
        deezer_data = enrich_with_deezer(row["title"], row["artist"])

        youtube_data = {"youtube_url": None, "youtube_embed": None}

        if not deezer_data["preview_url"]:
            youtube_data = enrich_with_youtube(row["title"], row["artist"])

        return {
            "title": row["title"],
            "artist": row["artist"],
            "similarity": float(row["similarity"]),
            "deezer_url": deezer_data["deezer_url"],
            "preview_url": deezer_data["preview_url"],
            "album_image": deezer_data["album_image"],
            "youtube_url": youtube_data["youtube_url"],
            "youtube_embed": youtube_data["youtube_embed"]
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        enriched_songs = list(executor.map(enrich_row, unique_songs))

    return {
        "model_version": MODEL_VERSION,
        "predicted_emotion": predicted_emotion,
        "songs": enriched_songs
    }
# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Linguistic Music Recommender API Running",
        "model_version": MODEL_VERSION
    })


@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend():

    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "No input text provided"}), 400

        result = recommend_songs(text)

        if result is None:
            return jsonify({"error": "Unable to process input text"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=8000)
