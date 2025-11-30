import pandas as pd
import numpy as np
import os

# Paths
RAW_DIR = "../dataset/raw"
OUT_DIR = "."
IMG_DIR = "images"

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

print("Loading data...")
anime = pd.read_csv(os.path.join(RAW_DIR, "anime.csv"))
rating = pd.read_csv(os.path.join(RAW_DIR, "rating.csv"))

# ==========================================
# Anime Cleaning (Basic/Worse Version)
# ==========================================
print("Cleaning Anime...")

# 1. Genre: Take only the first genre (Loss of information)
# Fill NA first
anime['genre'] = anime['genre'].fillna('Unknown')
# Split and take first
anime['genre_encoded'] = anime['genre'].apply(lambda x: x.split(',')[0].strip())

# 2. Type: Fill NA
anime['type'] = anime['type'].fillna('Unknown')

# 3. Episodes: Handle Unknown -> Median, then Binning
# Replace Unknown with NaN
anime['episodes'] = anime['episodes'].replace('Unknown', np.nan)
# Convert to numeric
anime['episodes'] = anime['episodes'].astype(float)
# Fill NaN with median
anime['episodes'] = anime['episodes'].fillna(anime['episodes'].median())

# Binning Logic for Episodes (Target Class in User's Example)
def bin_episodes(row):
    # If it's a movie/special type, categorize as such
    if row['type'] in ['Movie', 'Special', 'OVA', 'ONA', 'Music']:
        return 'Movie/Special'
    
    x = row['episodes']
    if x <= 13:
        return 'Short_Series'
    elif x <= 26:
        return 'Medium_Series'
    elif x <= 100:
        return 'Long_Series'
    else:
        return 'Very_Long_Series'

anime['episodes_encoded'] = anime.apply(bin_episodes, axis=1)

# 4. Anime Rating: Simple Binning
# Fill NaN
anime['rating'] = anime['rating'].fillna(anime['rating'].median())

def bin_anime_rating(x):
    if x < 6.0:
        return 'Low'
    elif x < 8.0:
        return 'Average'
    else:
        return 'High'

anime['anime_rating_encoded'] = anime['rating'].apply(bin_anime_rating)

# 5. Members: Simple Binning
def bin_members(x):
    if x < 10000:
        return 'Low'
    elif x < 100000:
        return 'Medium'
    else:
        return 'High'

anime['members_encoded'] = anime['members'].apply(bin_members)

# Select columns for anime-cleaned.csv
# Keep original IDs for merging, but output specific columns
anime_out = anime[['anime_id', 'name', 'genre_encoded', 'type', 'episodes_encoded', 'anime_rating_encoded', 'members_encoded']]
anime_out.to_csv("anime-cleaned.csv", index=False)
print("Saved anime-cleaned.csv")

# ==========================================
# Rating Cleaning (Basic/Worse Version)
# ==========================================
print("Cleaning Rating...")

# 1. User Rating: Handle -1 and Bin
# -1 in dataset often means "watched but not rated". We will treat it as a category "No_Rating".
def bin_user_rating(x):
    if x == -1:
        return 'No_Rating'
    elif x < 6:
        return 'Low'
    elif x < 8:
        return 'Average'
    else:
        return 'High'

rating['user_rating_encoded'] = rating['rating'].apply(bin_user_rating)

# Select columns for rating-cleaned.csv
rating_out = rating[['user_id', 'anime_id', 'user_rating_encoded']]
rating_out.to_csv("rating-cleaned.csv", index=False)
print("Saved rating-cleaned.csv")

# ==========================================
# Combined
# ==========================================
print("Merging...")
combined = pd.merge(rating_out, anime_out, on='anime_id', how='left')
# Drop rows where anime info might be missing (if any)
combined = combined.dropna()

combined.to_csv("combined.csv", index=False)
print("Saved combined.csv")

# ==========================================
# Visualization
# ==========================================
print("Generating Images...")
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=anime_out, x='anime_rating_encoded')
plt.title('Distribution of Anime Ratings (Encoded)')
plt.savefig(os.path.join(IMG_DIR, "anime_rating_dist.png"))
plt.close()

plt.figure(figsize=(10, 6))
sns.countplot(data=anime_out, x='episodes_encoded')
plt.title('Distribution of Episode Lengths (Encoded)')
plt.xticks(rotation=45)
plt.savefig(os.path.join(IMG_DIR, "episode_len_dist.png"))
plt.close()


# ==========================================
# ARFF Generation
# ==========================================
print("Generating ARFFs...")

def to_arff(df, filename, relation_name):
    # Filter out columns that are not useful for Weka (IDs, Names) if they exist
    # But for the specific requested files, we might need to be careful.
    # We will drop IDs for the ARFF to ensure Weka can run algorithms without crashing on high cardinality.
    
    cols_to_write = [c for c in df.columns if c not in ['anime_id', 'user_id', 'name']]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"@RELATION {relation_name}\n\n")
        
        for col in cols_to_write:
            # Check dtype
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Nominal
                unique_vals = df[col].unique()
                # Clean values for ARFF (remove quotes, handle spaces)
                clean_vals = []
                for v in unique_vals:
                    s = str(v)
                    # Escape single quotes
                    s = s.replace("'", "\\'")
                    # Quote if contains space or special chars
                    if ' ' in s or ',' in s or '{' in s or '}' in s or '?' in s:
                        s = f"'{s}'"
                    clean_vals.append(s)
                
                # Join
                vals_str = ",".join(clean_vals)
                f.write(f"@ATTRIBUTE {col} {{{vals_str}}}\n")
            else:
                # Numeric
                f.write(f"@ATTRIBUTE {col} NUMERIC\n")
        
        f.write("\n@DATA\n")
        
        # Write data
        # It's faster to convert to string and join
        # But we need to handle the quoting logic again.
        # Let's use a simpler approach for writing data
        
        for i, row in df[cols_to_write].iterrows():
            line_vals = []
            for col in cols_to_write:
                val = row[col]
                s = str(val)
                s = s.replace("'", "\\'")
                if ' ' in s or ',' in s or '{' in s or '}' in s or '?' in s:
                    s = f"'{s}'"
                line_vals.append(s)
            f.write(",".join(line_vals) + "\n")

# 1. anime-cleaned.arff
to_arff(anime_out, "anime-cleaned.arff", "anime_data")

# 2. rating-cleaned.arff
# Sampling rating because it's huge (7M rows)
rating_sampled = rating_out.sample(n=100000, random_state=42)
to_arff(rating_sampled, "rating-cleaned.arff", "rating_data")

# 3. combined-cleaned.arff
# User requested sample around 100k
combined_sampled = combined.sample(n=100000, random_state=42)
to_arff(combined_sampled, "combined-cleaned.arff", "combined_data")

print("Done.")
