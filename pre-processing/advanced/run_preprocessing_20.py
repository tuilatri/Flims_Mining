import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'dataset', 'raw')
OUTPUT_DIR = BASE_DIR
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

os.makedirs(IMAGES_DIR, exist_ok=True)

print("="*80)
print("PRE-PROCESSING-20: ANIME & RATING DATA PREPROCESSING (WITH VISUALIZATIONS)")
print("="*80)

# ============================================================================
# PART 1: LOAD DATA & BEFORE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("PART 1: LOADING DATA & BEFORE ANALYSIS")
print("="*80)

# 1. Load Data
anime_path = os.path.join(RAW_DIR, 'anime.csv')
rating_path = os.path.join(RAW_DIR, 'rating.csv')

print(f"Loading anime data from: {anime_path}")
anime_data = pd.read_csv(anime_path)
print(f"Loading rating data from: {rating_path}")
rating_data = pd.read_csv(rating_path)

# 2. Display Basic Info
print("\n--- Anime Data Overview ---")
print(anime_data.head())
print(anime_data.info())
print(anime_data.describe(include='all'))

print("\n--- Rating Data Overview ---")
print(rating_data.head())
print(rating_data.info())
print(rating_data.describe(include='all'))

# Statistics
print("\n--- Basic Statistics ---")
print(f"Anime Rows: {len(anime_data)}")
print(f"Rating Rows: {len(rating_data)}")
print(f"Anime Nulls:\n{anime_data.isnull().sum()}")
print(f"Rating Nulls:\n{rating_data.isnull().sum()}")

rating_minus_1_count = (rating_data['rating'] == -1).sum()
print(f"Rating = -1 Count: {rating_minus_1_count} ({rating_minus_1_count/len(rating_data)*100:.2f}%)")

# 3. Before Visualizations

# Histogram rating raw
plt.figure(figsize=(10, 6))
plt.hist(rating_data['rating'], bins=12, color='skyblue', edgecolor='black')
plt.title('Distribution of Raw Ratings (Before Processing)')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig(os.path.join(IMAGES_DIR, 'before_rating_hist.png'))
plt.close()
print("Saved: images/before_rating_hist.png")

# Pie chart rating = -1 vs valid
valid_ratings = len(rating_data) - rating_minus_1_count
plt.figure(figsize=(8, 8))
plt.pie([valid_ratings, rating_minus_1_count], labels=['Valid Ratings', 'Rating = -1'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Proportion of Valid vs -1 Ratings')
plt.savefig(os.path.join(IMAGES_DIR, 'before_rating_pie.png'))
plt.close()
print("Saved: images/before_rating_pie.png")

# Bar chart top genre before encoding
# Parse genres
all_genres = []
for genres in anime_data['genre'].dropna():
    all_genres.extend([g.strip() for g in str(genres).split(',')])
genre_counts = pd.Series(all_genres).value_counts().head(20)

plt.figure(figsize=(12, 8))
sns.barplot(y=genre_counts.index, x=genre_counts.values, palette='viridis')
plt.title('Top 20 Genres (Before Encoding)')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.savefig(os.path.join(IMAGES_DIR, 'before_genre_bar.png'))
plt.close()
print("Saved: images/before_genre_bar.png")


# ============================================================================
# PART 2: PREPROCESSING (V19 LOGIC)
# ============================================================================

print("\n" + "="*80)
print("PART 2: PREPROCESSING (V19 LOGIC)")
print("="*80)

anime_cleaned = anime_data.copy()
rating_cleaned = rating_data.copy()

# 2.1 Handle Missing Values (Anime)
anime_cleaned['genre'] = anime_cleaned['genre'].fillna('Unknown')
anime_cleaned['type'] = anime_cleaned['type'].fillna('Unknown')
anime_cleaned['episodes'] = anime_cleaned['episodes'].replace('Unknown', '0')
anime_cleaned['episodes'] = pd.to_numeric(anime_cleaned['episodes'], errors='coerce').fillna(0).astype(int)

anime_cleaned['rating'] = pd.to_numeric(anime_cleaned['rating'], errors='coerce')
median_rating = anime_cleaned['rating'].median()
anime_cleaned['rating'].fillna(median_rating, inplace=True)

anime_cleaned['members'] = anime_cleaned['members'].fillna(0).astype(int)

print("Missing values handled.")

# 2.2 Genre Encoding (TF-IDF + KMeans)
print("Encoding Genres (KMeans)...")
genre_data = anime_cleaned[['anime_id', 'genre']].copy()
genre_data['genre_cleaned'] = genre_data['genre'].str.replace(',', ' ')

vectorizer = TfidfVectorizer()
X_genre = vectorizer.fit_transform(genre_data['genre_cleaned'])

k_genre = 10
kmeans_genre = KMeans(n_clusters=k_genre, random_state=42, n_init=10)
genre_data['genre_encoded'] = kmeans_genre.fit_predict(X_genre)

anime_cleaned = anime_cleaned.merge(genre_data[['anime_id', 'genre_encoded']], on='anime_id', how='left')

# 2.3 Members Clustering
def encode_members(members):
    if members == 0: return 0
    elif 1 <= members <= 5000: return 1
    elif 5001 <= members <= 20000: return 2
    elif 20001 <= members <= 50000: return 3
    elif 50001 <= members <= 100000: return 4
    elif 100001 <= members <= 200000: return 5
    elif 200001 <= members <= 500000: return 6
    else: return 7

anime_cleaned['members_encoded'] = anime_cleaned['members'].apply(encode_members)

# 2.4 Anime Rating Clustering
def encode_rating(rating):
    if rating < 3.0: return 0
    elif 3.0 <= rating < 5.0: return 1
    elif 5.0 <= rating < 6.5: return 2
    elif 6.5 <= rating < 7.5: return 3
    elif 7.5 <= rating < 8.5: return 4
    elif 8.5 <= rating < 9.0: return 5
    else: return 6

anime_cleaned['anime_rating_encoded'] = anime_cleaned['rating'].apply(encode_rating)

# 2.5 Episodes Encoding
def encode_episodes(episodes):
    if episodes <= 1: return 'Movie/Special'
    elif 2 <= episodes <= 13: return 'Short_Series'
    elif 14 <= episodes <= 26: return 'Medium_Series'
    elif 27 <= episodes <= 100: return 'Long_Series'
    else: return 'Very_Long_Series'

anime_cleaned['episodes_encoded'] = anime_cleaned['episodes'].apply(encode_episodes)

# 2.6 User Rating Encoding
def encode_user_rating(rating):
    if rating == -1: return 0
    elif 1 <= rating <= 3: return 1
    elif 4 <= rating <= 5: return 2
    elif 6 <= rating <= 7: return 3
    elif 8 <= rating <= 9: return 4
    else: return 5

rating_cleaned['user_rating_encoded'] = rating_cleaned['rating'].apply(encode_user_rating)

# 2.7 Merge
print("Merging datasets...")
combined_cleaned = rating_cleaned.merge(
    anime_cleaned[['anime_id', 'genre_encoded', 'members_encoded', 'anime_rating_encoded', 'type', 'episodes_encoded']],
    on='anime_id',
    how='left'
)

# ============================================================================
# PART 3: AFTER VISUALIZATIONS & EXPORT
# ============================================================================

print("\n" + "="*80)
print("PART 3: AFTER ANALYSIS & EXPORT")
print("="*80)

# Display Cleaned Info
print("\n--- Anime Cleaned Overview ---")
print(anime_cleaned.head())
print(anime_cleaned.info())

print("\n--- Rating Cleaned Overview ---")
print(rating_cleaned.head())
print(rating_cleaned.info())

print("\n--- Combined Overview ---")
print(combined_cleaned.head())
print(combined_cleaned.info())

# Visualizations After

# Histogram rating_cleaned
plt.figure(figsize=(10, 6))
sns.countplot(x='user_rating_encoded', data=rating_cleaned, palette='coolwarm')
plt.title('Distribution of User Ratings (Encoded)')
plt.xlabel('Encoded Rating (0=No Rating, 1-5=Scale)')
plt.ylabel('Count')
plt.savefig(os.path.join(IMAGES_DIR, 'after_rating_hist.png'))
plt.close()
print("Saved: images/after_rating_hist.png")

# Bar chart genre encoded
plt.figure(figsize=(10, 6))
sns.countplot(x='genre_encoded', data=anime_cleaned, palette='viridis')
plt.title('Distribution of Genre Clusters (Encoded)')
plt.xlabel('Genre Cluster ID')
plt.ylabel('Count')
plt.savefig(os.path.join(IMAGES_DIR, 'after_genre_bar.png'))
plt.close()
print("Saved: images/after_genre_bar.png")

# Bar chart anime count after cleaning (by Type)
plt.figure(figsize=(10, 6))
sns.countplot(x='type', data=anime_cleaned, palette='Set2')
plt.title('Anime Count by Type (Cleaned)')
plt.xlabel('Type')
plt.ylabel('Count')
plt.savefig(os.path.join(IMAGES_DIR, 'after_anime_count.png'))
plt.close()
print("Saved: images/after_anime_count.png")

# User rating distribution
plt.figure(figsize=(10, 6))
valid_user_ratings = rating_cleaned[rating_cleaned['rating'] != -1]['rating']
plt.hist(valid_user_ratings, bins=10, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribution of Valid User Ratings (After Cleaning)')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig(os.path.join(IMAGES_DIR, 'after_user_rating_dist.png'))
plt.close()
print("Saved: images/after_user_rating_dist.png")


# Export CSV
print("\nExporting CSVs...")
anime_cleaned.to_csv(os.path.join(OUTPUT_DIR, 'anime-cleaned.csv'), index=False)
rating_cleaned.to_csv(os.path.join(OUTPUT_DIR, 'rating-cleaned.csv'), index=False)
combined_cleaned.to_csv(os.path.join(OUTPUT_DIR, 'combined.csv'), index=False)
print("CSVs exported.")

# Export ARFF
print("Exporting ARFFs...")

def create_arff_file(df, filename, relation_name, exclude_cols=[]):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"@relation {relation_name}\n\n")
        cols_to_write = [c for c in df.columns if c not in exclude_cols]
        for col in cols_to_write:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                escaped_vals = [str(v).replace("'", "\\'").replace('"', '\\"') for v in unique_vals if pd.notna(v)]
                if len(escaped_vals) > 500: escaped_vals = escaped_vals[:500]
                vals_str = ','.join([f"'{v}'" for v in escaped_vals])
                f.write(f"@attribute {col} {{{vals_str}}}\n")
            elif df[col].dtype in ['int64', 'int32']:
                unique_vals = sorted(df[col].unique())
                vals_str = ','.join([str(int(v)) for v in unique_vals if pd.notna(v)])
                f.write(f"@attribute {col} {{{vals_str}}}\n")
            elif df[col].dtype in ['float64', 'float32']:
                unique_vals = sorted(df[col].unique())
                vals_str = ','.join([f"{v:.2f}" for v in unique_vals if pd.notna(v)])
                f.write(f"@attribute {col} {{{vals_str}}}\n")
        f.write("\n@data\n")
        for _, row in df.iterrows():
            row_data = []
            for col in cols_to_write:
                val = row[col]
                if pd.isna(val): row_data.append('?')
                elif df[col].dtype == 'object':
                    escaped_val = str(val).replace("'", "\\'").replace('"', '\\"')
                    row_data.append(f"'{escaped_val}'")
                elif df[col].dtype in ['int64', 'int32']:
                    row_data.append(str(int(val)))
                elif df[col].dtype in ['float64', 'float32']:
                    row_data.append(f"{val:.2f}")
                else:
                    row_data.append(str(val))
            f.write(','.join(row_data) + '\n')
    print(f"Saved: {filename}")

create_arff_file(anime_cleaned, os.path.join(OUTPUT_DIR, 'anime-cleaned.arff'), 'anime_data')

# Sample for rating and combined to avoid huge files
rating_sample = rating_cleaned.sample(n=min(10000, len(rating_cleaned)), random_state=42)
create_arff_file(rating_sample, os.path.join(OUTPUT_DIR, 'rating-cleaned.arff'), 'rating_data')

combined_sample = combined_cleaned.sample(n=min(10000, len(combined_cleaned)), random_state=42)
cols_to_exclude = ['user_id', 'anime_id', 'rating']
create_arff_file(combined_sample, os.path.join(OUTPUT_DIR, 'combined-cleaned.arff'), 'combined_data', exclude_cols=cols_to_exclude)

print("\nProcessing Complete.")
