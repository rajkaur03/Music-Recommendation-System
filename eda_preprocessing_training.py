### STEP 1: set up environment
import zipfile
import json
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import time
from spotipy.exceptions import SpotifyException
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from lightgbm import LGBMRanker
import pickle
import shap
from statistics import mean
import os

path = 'user_signin/data/'
if os.path.isdir(f"../{path}"):
    print("Path Exists!")
else:
    os.mkdir(f"../{path}")
    
model_path = 'user_signin/pkl_models/'
if os.path.isdir(f"../{model_path}"):
    print("Path Exists!")
else:
    os.mkdir(f"../{model_path}")
    
client_credentials_manager = SpotifyClientCredentials(client_id='2f215c1cac614d99b3163fd183786419', client_secret='ab72ae8d16344084b8e9d1d8699340ef')

sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

### STEP 2: read in Spotify Million Playlist Dataset

# Load playlist data from JSON file
playlist_names = []
playlist_songs = []
unique_tracks = set()
track_names = {}
track_albums = {}
# Iterate through files in the folder
z = zipfile.ZipFile(path+"spotify_million_playlist_dataset.zip", "r")
for filename in z.namelist():
    if filename.endswith('.json'):  # Check if the file is a JSON file            
        with z.open(filename) as f:  
            data = f.read()  
            playlist_data = json.loads(data) 
            for playlist in playlist_data["playlists"]:
                playlist_names.append(playlist["name"])
                playlist_songs.append([song["track_uri"].replace("spotify:track:", "") for song in playlist["tracks"]])
                for track in playlist['tracks']:
                    track_uri = track['track_uri']
                    track_name = track['track_name']
                    album_uri = track.get('album_uri', '')
                    track_names[track_uri] = track_name
                    track_albums[track_uri] = album_uri
                    unique_tracks.add(track_uri)
# Create DataFrame
million_playlist = pd.DataFrame({
    "playlist_name": playlist_names,
    "songs": playlist_songs
})

million_playlist.to_csv(path+"million_playlist.csv")
#Returns csv with playlist name (str) and songs column (str)

### STEP 3: Get song features for each song in Spotify Million Playlist Dataset
class RateLimiter:
    def __init__(self, max_requests, period):
        self.max_requests = max_requests
        self.period = period
        self.requests = []

    def wait(self):
        now = time.time()
        while self.requests and self.requests[0] + self.period < now:
            self.requests.pop(0)
        if len(self.requests) >= self.max_requests:
            sleep_time = self.period - (now - self.requests[0])
            if sleep_time > 0:
                print(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.wait()
        self.requests.append(time.time())


rate_limiter = RateLimiter(100, 60)

def get_audio_features(track_id):
    print(f"Requesting audio features for track ID: {track_id}")

    # Wait for 2 seconds before making the request
    time.sleep(2)

    try:
        audio_features = sp.audio_features(track_id)[0]
        if audio_features:
            print(f"Audio features for track ID {track_id}: {audio_features}")
            return {
                'danceability': float(audio_features['danceability']),
                'energy': float(audio_features['energy']),
                'key': float(audio_features['key']),
                'loudness': float(audio_features['loudness']),
                'mode': float(audio_features['mode']),
                'speechiness': float(audio_features['speechiness']),
                'acousticness': float(audio_features['acousticness']),
                'instrumentalness': float(audio_features['instrumentalness']),
                'liveness': float(audio_features['liveness']),
                'valence': float(audio_features['valence']),
                'tempo': float(audio_features['tempo']),
            }
    except SpotifyException as e:
        if e.http_status == 429:
            retry_after = int(e.headers.get('Retry-After', 1))
            print(f"Rate limit reached, retrying after {retry_after} seconds")
            time.sleep(retry_after)
            return get_audio_features(track_id)
        else:
            print(f"Error requesting audio features for track ID {track_id}: {e}")
            raise e
    except Exception as e:
        print(f"An unexpected error occurred while requesting audio features for track ID {track_id}: {e}")

    return {
        'danceability': 0,
        'energy': 0,
        'key': 0,
        'loudness': 0,
        'mode': 0,
        'speechiness': 0,
        'acousticness': 0,
        'instrumentalness': 0,
        'liveness': 0,
        'valence': 0,
        'tempo': 0,
    }

def insert_tracks_to_db(track_uris,track_names,track_albums):
    song_list = []
    for uri in track_uris:
        actual_uri = uri.split(':')[2]
        actual_album_uri = track_albums[uri].split(':')[2] if ':' in track_albums[uri] else track_albums[uri]
        track_name = track_names[uri][:255]
        audio_features = get_audio_features(actual_uri)  # Fetch audio features
        # Prepare the data tuple including audio features
        data = (
            actual_uri, track_name, actual_album_uri, audio_features['danceability'], audio_features['energy'], audio_features['key'],
            audio_features['loudness'], audio_features['mode'], audio_features['speechiness'],
            audio_features['acousticness'], audio_features['instrumentalness'],
            audio_features['liveness'], audio_features['valence'], audio_features['tempo']
        )
        song_list.append(data)
    song_features = pd.DataFrame(song_list,columns=['track_uri', 'track_name', 'album_uri', 'danceability', 'energy', 'key_feature', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])
    return song_features

# Example usage
song_features = insert_tracks_to_db(unique_tracks,track_names,track_albums)
song_features.to_csv(path+"song_features.csv")
### STEP 4: EDA 
song_features.head()

song_features.count()

# checking size of data
song_features.shape

# checking null value
pd.isnull(song_features).sum()

# concise summary of the DataFrame
song_features.info()

# Descriptive statistics of the numerical variables present in columns
song_features.describe().transpose()

# Function to plot histograms
def plot_histograms(data):
    numerical_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numerical_columns, start=1):
        plt.subplot(3, 3, i)
        sns.histplot(data[column], bins=20, kde=True)
        plt.title(column.capitalize())
    plt.tight_layout()
    plt.show()

# Function to plot boxplots
def plot_boxplots(data):
    numerical_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numerical_columns, start=1):
        plt.subplot(3, 3, i)
        sns.boxplot(data[column])
        plt.title(column.capitalize())
    plt.tight_layout()
    plt.show()

# Function to plot bar charts
def plot_bar_charts(data):
    plt.figure(figsize=(12, 4))
    categorical_columns = ['key_feature', 'mode']
    for i, column in enumerate(categorical_columns, start=1):
        plt.subplot(1, 2, i)
        data[column].value_counts().plot(kind='bar')
        plt.title(column.capitalize())
    plt.tight_layout()
    plt.show()

# Function to plot scatter plot (pairplot)
def plot_scatter_pairplot(data):
    sns.pairplot(data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']])
    plt.show()

# Function to plot violin plots
def plot_violin_plots(data):
    numerical_columns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numerical_columns, start=1):
        plt.subplot(3, 3, i)
        sns.violinplot(data[column])
        plt.title(column.capitalize())
    plt.tight_layout()
    plt.show()

# Call the functions to plot the visualizations
plot_histograms(song_features)
plot_boxplots(song_features)
plot_bar_charts(song_features)
plot_scatter_pairplot(song_features)
plot_violin_plots(song_features)

#top 10 least energetic songs
least_energetic = song_features.sort_values('energy', ascending = True).head(10)
least_energetic[['track_name','album_uri']]

#top 10 most energetic songs
most_energetic = song_features.sort_values('energy', ascending = False).head(10)
most_energetic[['track_name','album_uri']]

#visualize through correlation map
corr_df=song_features.corr(method='pearson')
plt.figure(figsize=(14,8))
heatmap=sns.heatmap(corr_df, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap="YlGnBu", linewidths=2, linecolor="Black"
)
heatmap.set_title('Correlation Heatmap Between Variable')
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90);

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Regression plot between Loudness and Energy
sns.regplot(data=song_features, ci=None, y='loudness', x='energy', scatter_kws={"color": "blue", 's': 2}, marker='2', line_kws={"color": "black", 'linewidth': 1.5}, ax=axes[0, 0]).set(title='Loudness Vs Energy Correlation')

# Regression plot between Valence and Acousticness
sns.regplot(data=song_features, ci=None, y='valence', x='acousticness', scatter_kws={"color": "green", 's': 3}, marker='X', line_kws={"color": "black", 'linewidth': 1.5}, ax=axes[0, 1]).set(title='Valence Vs Acousticness Correlation')

# Regression plot between Speechiness and Acousticness
sns.regplot(data=song_features, ci=None, y='speechiness', x='acousticness', scatter_kws={"color": "orange", 's': 3}, marker='+', line_kws={"color": "black", 'linewidth': 1.5}, ax=axes[1, 0]).set(title='Speechiness Vs Acousticness Correlation')

# Regression plot between Valence and Danceability
sns.regplot(data=song_features, ci=None, y='valence', x='danceability', scatter_kws={"color": "purple", 's': 3}, marker='*', line_kws={"color": "black", 'linewidth': 1.5}, ax=axes[1, 1]).set(title='Valence Vs Danceability Correlation')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

### STEP 5: cluster songs

# Define the specific cluster values to try
cluster_values = [50, 70, 100, 120, 150, 170, 200]

wcss = []  # Within-cluster sum of squares

# Iterate over the specified cluster values
number_cols = ['danceability','energy', 'key_feature', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X = song_features[number_cols].select_dtypes(np.number)
for n_clusters in cluster_values:
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    # Calculate the WCSS and append to the list
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(cluster_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
# Add annotations for WCSS values
for i, txt in enumerate(wcss):
    plt.annotate(np.round(txt, 2), (cluster_values[i], wcss[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=100, random_state=42, verbose=False))], verbose=False)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data = song_features.copy()
data['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['track_name']
projection['cluster'] = data['cluster_label']

fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'], color_discrete_sequence=px.colors.qualitative.Pastel1)
fig.show();

projection.head(50)

song_cluster = data[['track_uri', 'cluster_label']]
song_cluster.to_csv(path+'song_cluster.csv',index=False)

### STEP 6: cluster playlists
def calculate_embedding(songs):
    song_cluster_mapping = dict(zip(song_cluster['track_uri'], song_cluster['cluster_label']))
    embedding = [0] * 100
    total_songs = 0
    for song in songs:
        cluster_label = song_cluster_mapping.get(song)
        if cluster_label is not None:
            embedding[cluster_label] += 1
            total_songs += 1
    if total_songs > 0:
        for i in range(0, 100):
            embedding[i] /= total_songs # Normalisation at playlist level
    return embedding

df = million_playlist.copy()
df['embedding'] = df['songs'].apply(calculate_embedding)

# Define the range of max_clusters values
min_clusters = 10
max_clusters = 50
step = 10
cluster_values = list(range(min_clusters, max_clusters + 1, step))

wcss = []  # Within-cluster sum of squares
for i in range(100):
   df[f'd_{i+1}'] = df['embedding'].apply(lambda x: x[i])
number_cols = []
for i in range(1, 101):
    number_cols.append(f'd_{i}')
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=35, verbose=False))], verbose=False)
X = df[number_cols].select_dtypes(np.number)
# Iterate over the specified cluster values
for n_clusters in cluster_values:
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    # Calculate the WCSS and append to the list
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(cluster_values, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)

# Add annotations for WCSS values
for i, txt in enumerate(wcss):
    plt.annotate(np.round(txt, 2), (cluster_values[i], wcss[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=35, random_state = 42, verbose=False))], verbose=False)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
df['cluster_label'] = song_cluster_labels

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = df['playlist_name']
projection['cluster'] = df['cluster_label']

fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show();

playlist_cluster = df[['playlist_name', 'embedding', 'cluster_label']]
playlist_cluster.to_csv(path+'playlist_cluster.csv',index=False)

### STEP 7: collaboratively filter using songs within each playlist cluster label
#form df with songs per playlist cluster label for collaborative filtering
clusters = playlist_cluster['cluster_label'].unique()
cluster_2_song = pd.DataFrame(columns=['cluster_label','track_uri'])
#for every cluster
for i in clusters:
    cluster_playlist = playlist_cluster[playlist_cluster['cluster_label']==i]
    #million playlist and cluster playlist_cluster must be in the same order
    merge = pd.merge(cluster_playlist, million_playlist, left_index=True, right_index=True)
    cluster_songs=[]
    #for every playlist in cluster
    for j in range(0,len(cluster_playlist)):
        playlist_songs = merge.iloc[j]['songs']
        #for every song in playlist
        for k in playlist_songs:
            cluster_songs.append(k)
    cluster_songs =list(set(cluster_songs))
    cluster_col = [i]*len(cluster_songs)
    combined = {'cluster_label':cluster_col,'track_uri':cluster_songs}
    cluster_songs = pd.DataFrame(combined)
    cluster_2_song = pd.concat([cluster_2_song,cluster_songs])
    
cluster_2_song.to_csv(path+'cluster_2_song.csv',index=False)

### STEP 8: create training data and train metric learning to rank

#get 10 ranked songs per playlist
def get_ranked_songs(s,song_features):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import math
    
    def get_euc(number_cols,agg_data,s):
        median_data = agg_data[agg_data['cluster_label']==s.loc['cluster_label']][number_cols]
        list1 = list(s[number_cols])
        list2 = list(median_data.iloc[0])
        euc_dis = math.dist(list1,list2)
        return euc_dis
    
    number_cols = ['danceability','energy', 'key_feature', 'loudness', 'mode', 'speechiness',
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    playlist_songs = s[2]
    playlist_songs_data = song_features[song_features['track_uri'].isin(playlist_songs)]
    playlist_songs_features = playlist_songs_data[number_cols].select_dtypes(np.number)
    if(len(playlist_songs_features)>10):
        kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, verbose=False, random_state=42,n_init='auto'))], verbose=False)
        np_array = playlist_songs_features.to_numpy()
        kmeans.fit(np_array)
        song_cluster_labels = kmeans.predict(np_array)
        playlist_songs_data['cluster_label'] = song_cluster_labels
        scaler = StandardScaler() 
        playlist_songs_data[number_cols]  = scaler.fit_transform(playlist_songs_data[number_cols]) 
        agg_data = playlist_songs_data.groupby('cluster_label')[number_cols].median().reset_index()
        playlist_songs_data["euc_dis"] = playlist_songs_data.apply(lambda row: get_euc(number_cols,agg_data,row),axis=1)
        cluster_songs = playlist_songs_data.loc[playlist_songs_data.groupby('cluster_label').euc_dis.idxmin()]
        cluster_count = playlist_songs_data.groupby('cluster_label')['track_uri'].count()
        cluster_songs['count_num'] = list(cluster_count)
        cluster_songs['playlist_name'] = s[1]
        #add playlist_id and rank
        j=1;
        playlist_data = cluster_songs.sort_values('count_num').reset_index(drop=True)
        playlist_data['playlist_id']= s[0]
        rank = [j]
        for k in range(1,10):
            if playlist_data['count_num'][k] > playlist_data['count_num'][k-1]:
                j+=1
            rank.append(j)
        playlist_data['rank_num'] = rank
        output = playlist_data[['playlist_name','playlist_id', 'track_uri','cluster_label','count_num','euc_dis','rank_num']]
        return output
    
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

#test = million_playlist[0:4]
#series_test = test.parallel_apply(get_ranked_songs, song_features=song_features, axis=1)
#df_test = pd.concat(list(series_test))

# Set system environment variable OMP_NUM_THREADS = 1 and restart computer. For 6 core computer, this required at least 20 GB of RAM and took 10 hours to run.
parallelized_data = million_playlist.parallel_apply(get_ranked_songs, song_features=song_features, axis=1)
playlist_rank_data = pd.concat(list(parallelized_data))
playlist_rank_data.to_csv(path+'playlist_rank_data.csv',index=False)  

#Train Metric Learning to Rank______________________________________________________________________________|
#https://towardsdatascience.com/how-to-implement-learning-to-rank-model-using-python-569cd9c49b08
features =  ['danceability','energy', 'key_feature', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'rank_num'

training_with_features = playlist_rank_data.merge(song_features,how='left',on='track_uri')
training_with_features.sort_values(by='playlist_id',inplace=True)
training_with_features.set_index('playlist_id',inplace=True)
test_size = int(len(training_with_features)*0.2)
X,y = training_with_features[features],training_with_features[target]
test_idx_start = len(X)-test_size

xtrain,xtest,ytrain,ytest = X.iloc[0:test_idx_start],X.iloc[test_idx_start:],y.iloc[0:test_idx_start],y.iloc[test_idx_start:]

get_group_size = lambda df: df.reset_index().groupby("playlist_id")['playlist_id'].count()

train_groups = get_group_size(xtrain)
test_groups = get_group_size(xtest)

ranker = LGBMRanker(objective="lambdarank")
ranker.fit(xtrain,ytrain,group=train_groups,eval_set=[(xtest,ytest)],eval_group=[test_groups],eval_metric=['ndcg'])
results = ranker.evals_result_
results = results['valid_0']['ndcg@1']+results['valid_0']['ndcg@2']+results['valid_0']['ndcg@3']+results['valid_0']['ndcg@4']+results['valid_0']['ndcg@5']
avg_ndcg = mean(results)
min_ndcg = min(results)
max_ndcg = max(results)
ranker.feature_importances_

with open(path+"MLR.pkl", "wb") as f:
    pickle.dump(ranker, f)
    
#Generate Feature Importance Plots
X_train = xtrain
feature_names = features
explainer = shap.Explainer(ranker, X_train, feature_names=feature_names)
shap_values = explainer(X_train,check_additivity=False)

# Create a figure with 2 subplots
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

# Plot the summary plot on the first subplot
plt.tight_layout()
plt.subplot(1, 2, 1)
shap.summary_plot(shap_values, feature_names=feature_names, plot_type='bar')

# Plot the feature importance plot on the second subplot
plt.subplot(1, 2, 2)
shap.summary_plot(shap_values, feature_names=feature_names, plot_type='dot')

### STEP 9: create table to hold future predictions 
#Create base predictions database to add to for future reference of what predictions were made
predictions_full_data = pd.DataFrame(columns=['playlist_name','playlist_id','track_uri','score'])
predictions_full_data.to_csv(path+'predictions_full_data.csv',index=False)