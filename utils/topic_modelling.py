import numpy as np
import pandas as pd
import pylab as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

'''
TOPIC MODELLING
'''

# cluster mapping
cluster_labels = {
    0: 'Answers',
    1: 'Control',
    2: 'Entertainment',
    3: 'Timers',
    4: 'Weather',
    5: 'Undefined',
    6: 'Attempt questionnaire',
    7: 'Reminders/Time/Date',
    8: 'News',
    9: 'Greetings'
    }

def remove_words(input):
    """
    Preprocessing: Remove stop words 
    """
    addstop = ['alexa', 'echo']
    wordlist = input.split()
    new = [x for x in wordlist if x not in addstop]
    new = " ".join(new)
    return new

def sentence_embeddings(df):
    """
    Create sentence embeddings using pretrained model 
    """
    model = SentenceTransformer('all-mpnet-base-v2')
    data = df.user_said
    embeddings = model.encode(data, show_progress_bar=True)
    df_new = df.copy()
    df_new['embeddings']=pd.Series(list(embeddings))
    return df_new
    
def create_df_embed(df):
    """
    Usage: filter data and create df with embeddings
    """
    df_random = df[df['int_type'] == 'r'].copy()
    df_random.reset_index(inplace = True, drop=True)
    df_random['user_said'] = df_random['user_said'].apply(lambda x: remove_words(x))
    df_random['user_said'] = [
        x.replace('b. b. c.', 'bbc').\
        replace('i. t. v.', 'tv').\
            replace('t. v.', 'tv').\
                replace('f. m.', 'fm').\
                    replace('l. b. c.', 'lbc')\
                    for x in df_random['user_said']
                    ]
    df_embed = sentence_embeddings(df_random)
    return df_embed

def cluster_kmeans(n_clusters, embed):
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_model.fit(X=embed)
    clusters = cluster_model.labels_
    return clusters 

def cluster_agglom(n_clusters, embed):
    corpus_embeddings = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    cluster_agglom = AgglomerativeClustering(n_clusters=n_clusters) 
    cluster_agglom.fit(corpus_embeddings)
    clusters = cluster_agglom.labels_
    return clusters

def clustered(clusters, df):
    """
    Usage: Create list of preprocessed sentences for each cluster computed 
    """
    clustered_sentences = [[] for i in range(max(clusters.tolist())+1)]
    for sentence_id, cluster_id in enumerate(clusters):
        clustered_sentences[cluster_id].append(df.user_said[sentence_id])
    return clustered_sentences

def get_topics(df, clustering):
    """
    Returns updated df with topic for each sentence embedding 
    for a given dataset and clustering method
    Arguments:
    - df: The dataframe with ransformed documents to embeddings (df.embeddings)
    - clustering: Sets the clustering method to use on sentence embeddings 
    Notes: 
    - skip the dimensionality reduction step as kmeans can handle high dimensionality like a cosine-based kmeans.
    """
    print(f'Clustering method used: {clustering}')
    # resize needed to compute silhouette score
    embed = [] 
    for i in np.array(df.embeddings):
        embed.append(list(i))
    best_score = 0 
    for num in range(6,18): 
        clusters = cluster_kmeans(num, embed) if clustering == 'kmeans' else cluster_agglom(num, embed) 
        score = silhouette_score(embed, clusters, metric='euclidean')
        print(f'{num} topics: silhouette score = {score}')
        if score > best_score:
            best_num_clusters = num
            best_score = score
    # create clusters with best number of topics returned by silhouette score 
    clusters = cluster_kmeans(best_num_clusters, embed)
    # add column with predicted cluster
    print('--->')
    print(f'The silhouette score is the highest ({best_score}) for {best_num_clusters} topics.')
    df_new = df.copy()
    df_new['pred'] = pd.Series(clusters)
    # create dict with topic and frequency in df
    topic = np.unique(clusters)
    count = {}
    for i in topic:
        freq = len(df_new.loc[clusters == i])
        count[i] = freq
    # sentences = clustered(clusters, df)
    # return df_new, count, sentences
    return df_new, count

def get_tsne(df,n_components):
    emb = np.array(df.embeddings.tolist())
    transformer = TSNE(n_components=n_components, perplexity=50.0, early_exaggeration=12.0, 
                       learning_rate=40, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, 
                       metric='euclidean', metric_params=None, init='pca', random_state=None, 
                       method='barnes_hut', angle=0.5)
    emb_transformed = transformer.fit_transform(emb)
    
    df['transformed'] = [*emb_transformed]
    return df

# plot 3D if needed
def plot3d(df):
    fig = plt.figure(figsize=(16,12))
    ax = fig.add_subplot(projection='3d')
    for p in np.sort(df.pred.unique()):
        x = df.loc[df.pred == p]
        x_emb = np.array(x.transformed.tolist())#.reshape(len(x.transformed),3)
        ax.scatter(xs=x_emb[:,0], ys=x_emb[:,1], zs=x_emb[:,2], label=cluster_labels[p])
    ax.set_xlabel('dimension 0')
    ax.set_ylabel('dimension 1')
    ax.set_zlabel('dimension 2')
    ax.legend()
    plt.show()

# plot 2D if needed
def plot2d(df,dim1,dim2):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    for p in np.sort(df.pred.unique()):
        x = df.loc[df.pred == p]
        x_emb = np.array(x.transformed.tolist())#.reshape(len(x.transformed),3)
        ax.scatter(x_emb[:,dim1], x_emb[:,dim2], label=cluster_labels[p])
    ax.set_xlabel(f'dimension {dim1}')
    ax.set_ylabel(f'dimension {dim2}')
    ax.legend()
    plt.show()