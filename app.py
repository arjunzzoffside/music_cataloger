import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.utils
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- HELPER: AUTO-TAGGING LOGIC ---
def generate_vibe_tags(row):
    tags = []
    # Heuristics based on typical Spotify audio feature ranges
    if row.get('energy', 0) > 0.7: tags.append("High Energy")
    if row.get('energy', 0) < 0.4: tags.append("Chill")
    if row.get('acousticness', 0) > 0.7: tags.append("Acoustic")
    if row.get('danceability', 0) > 0.7: tags.append("Danceable")
    if row.get('instrumentalness', 0) > 0.6: tags.append("Instrumental")
    if row.get('valence', 0) > 0.7: tags.append("Happy")
    elif row.get('valence', 0) < 0.3: tags.append("Melancholic")
    if row.get('speechiness', 0) > 0.3: tags.append("Spoken/Rap")
    
    return ", ".join(tags) if tags else "Mixed Pop/Vibe"

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
        file.save(filepath)
        return redirect(url_for('analyze'))
    return render_template('index.html')

@app.route('/analyze')
@app.route('/analyze')
@app.route('/analyze')
def analyze():
    print("--- 1. STARTING ANALYSIS ---")
    
    # 1. Load Data
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    if not os.path.exists(filepath):
        print("Error: File not found.")
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    print(f"--- Loaded CSV: {len(df)} rows found. ---")  # <--- CHECK THIS NUMBER IN TERMINAL

    # 2. Preprocess
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    # Filter
    df_clean = df.dropna(subset=features).copy()
    print(f"--- After Cleaning: {len(df_clean)} rows remaining. ---") 
    
    # SAFETY CHECK: If we have less than 10 songs, stop immediately.
    if len(df_clean) < 10:
        return f"Error: You only have {len(df_clean)} valid songs. We need at least 10 to find 10 clusters. Check your CSV file."

    # OPTIONAL: Limit to 5000 songs to keep it fast (BUT NOT 1!)
    if len(df_clean) > 5000:
        df_clean = df_clean.sample(n=5000, random_state=42)
        print("--- Downsampled to 5000 songs for speed. ---")

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    
    # 3. Clustering (K-Means)
    num_clusters = 10 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # !!! THIS IS WHERE THE ERROR WAS HAPPENING !!!
    df_clean['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Generate Auto-Tags
    cluster_centers = df_clean.groupby('cluster')[features].mean()
    cluster_centers['tag'] = cluster_centers.apply(generate_vibe_tags, axis=1)
    tag_map = cluster_centers['tag'].to_dict()
    df_clean['vibe_tag'] = df_clean['cluster'].map(tag_map)
    df_clean['cluster_name'] = "Cluster " + df_clean['cluster'].astype(str) + ": " + df_clean['vibe_tag']
    
    # 5. Outlier Detection
    distances = kmeans.transform(X_scaled)
    min_distances = np.min(distances, axis=1)
    df_clean['outlier_score'] = min_distances
    threshold = np.percentile(min_distances, 95)
    df_clean['is_outlier'] = df_clean['outlier_score'] > threshold
    
    # 6. Visualization Data (PCA)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_clean['x'] = components[:, 0]
    df_clean['y'] = components[:, 1]
    
    # Create Graph
    fig = px.scatter(df_clean, x='x', y='y', color='cluster_name', 
                     hover_data=['name', 'artists'], 
                     title='Music Library Map', height=600)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Save & Return
    df_clean.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'organized_library.csv'), index=False)
    
    songs_data = df_clean.head(100).to_dict(orient='records')
    outliers_data = df_clean[df_clean['is_outlier']].sort_values('outlier_score', ascending=False).head(50).to_dict(orient='records')
    
    return render_template('dashboard.html', 
                           graphJSON=graphJSON, 
                           songs=songs_data, 
                           outliers=outliers_data,
                           cluster_map=tag_map)
    print("--- STARTING ANALYSIS ---")
    
    # 1. Load Data
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
    df = pd.read_csv(filepath)
    print(f"Original CSV size: {len(df)} rows") # DEBUG PRINT
    
    # 2. Preprocess
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    
    # Filter
    df_clean = df.dropna(subset=features).copy()
    print(f"Size after dropna: {len(df_clean)} rows") # DEBUG PRINT
    
    # --- SAFETY CHECK ---
    if len(df_clean) < 10:
        return "Error: Not enough data! We need at least 10 songs to find 10 clusters. Check your CSV columns."

    # Limit for performance (Ensure this is 1000, not 1)
    df_clean = df_clean.head(1000)
    print(f"Size sending to AI: {len(df_clean)} rows") # DEBUG PRINT

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    
    # ... Continue with the rest of your code ...
@app.route('/download')
def download():
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], 'organized_library.csv'), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)