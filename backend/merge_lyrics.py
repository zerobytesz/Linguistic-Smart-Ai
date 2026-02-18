import pandas as pd
import os

folder_path = "/Users/raghavtyagi/Downloads/linguistic-music-recommender/backend/csv"

all_data = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder_path, file))
        df.columns = [col.lower().strip() for col in df.columns]

        if "title" in df.columns and "lyric" in df.columns:
            temp = df[["title", "lyric", "artist"]].copy()
            temp.columns = ["title", "lyrics", "artist"]

            temp = temp.dropna()
            temp = temp[temp["lyrics"].str.len() > 100]

            all_data.append(temp)

final_df = pd.concat(all_data, ignore_index=True)
final_df = final_df.drop_duplicates(subset=["lyrics"])

final_df.to_csv("data/lyrics_train_ready.csv", index=False)

print("Merged successfully!")
print("Total songs:", len(final_df))
