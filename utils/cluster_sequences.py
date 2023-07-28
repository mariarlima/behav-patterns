import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
from collections import Counter
from kneed import KneeLocator
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.colors as mcolors


'''
CLUSTERING OF ACTIVITY SEQUENCES AND ANALYSIS OF SEQUENCES WITHIN CLUSTERS
'''

# events used for activity sequences
events = ['Bed_in', 'Bed_out', 'vitals', 'Bathroom', 'Bedroom', 'Hallway',
       'Lounge', 'Kitchen', 'Back door', 'Front door', 'r', 's_start',
       's_end']

def find_n_clusters(ds, random_state):
    sse = []
    for k in np.arange(2,15):
        sc = KMedoids(n_clusters=k, metric='precomputed', init='k-medoids++', random_state=random_state).fit(ds) 
        sse.append([k,silhouette_score(ds, sc.labels_)])
    return sse

def check_silhouette(sse):
    sse = pd.DataFrame(sse,columns=['k','silhouette'])
    kneedle = KneeLocator(sse.silhouette, sse.k, S=1.0, curve="convex", direction="decreasing")
    ax = sse.set_index('k').silhouette.plot()
    ax.vlines(sse[sse.silhouette==kneedle.knee].k,*sse.silhouette.agg(['min','max']),color='k',linestyle=':')
    ax.set_ylim(*sse.silhouette.agg(['min','max']))
    ax.set_xlim(*sse.k.agg(['min','max']))
    return kneedle

def apply_clustering(ds, random_state, a=True, k_choice=3):
    sse = find_n_clusters(ds, random_state)
    sse = pd.DataFrame(sse,columns=['k','silhouette'])
    kneedle = KneeLocator(sse.silhouette, sse.k, S=1.0, curve="convex", direction="decreasing")
    n = sse[sse.silhouette==kneedle.knee] if a else sse[sse.k==k_choice]
    print('K clusters used:', n)
    print('Check:', n.k.iat[0])
    sc = KMedoids(n_clusters=n.k.iat[0], metric='precomputed', init='k-medoids++',random_state=random_state).fit(ds) 
    return sc

def apply_clustering_2(ds, random_state, a=True, k_choice=3):
    sse = find_n_clusters(ds, random_state)
    sse = pd.DataFrame(sse,columns=['k','silhouette'])
    kneedle = KneeLocator(sse.silhouette, sse.k, S=1.0, curve="convex", direction="decreasing")
    n = sse[sse.silhouette==kneedle.knee] if a else sse[sse.k==k_choice]
    print('K clusters used:', n)
    sc = KMedoids(n_clusters=n.k.iat[0], metric='precomputed', init='k-medoids++',random_state=random_state).fit(ds) 
    return sc

def plot_clusters(ds, sc): # these two arguments need to be separate in case I want to define n_clusters manually
    from matplotlib.colors import LinearSegmentedColormap
    fig,ax = plt.subplots(1,2,figsize=(10,5),gridspec_kw={'width_ratios': [.70, .30]})
    sns.heatmap(ds, ax=ax[0])
    myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
    cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))
    sns.heatmap(sc.labels_.reshape(-1,1),cmap='viridis', cbar=False, ax=ax[1], xticklabels=False)
    return 

# to create contingency table
def show_gradient(df):
    df = df.apply(pd.to_numeric, errors='coerce', downcast=None)
    return df.style.background_gradient(cmap='Blues').format(precision=2)

def probability_sensor_trigger(df, random=True):
    # we are interested in the events before the predefined sink event ('r')
    data = df.transition.apply(lambda x: x[:-2]).apply(lambda x: x.split('>')) if random else df.transition.apply(lambda x: x[:-8]).apply(lambda x: x.split('>'))
    l = []
    for i in range(len(data)):
        l += data.iloc[i]
    counts = Counter(l)
    return counts

def index_clusters(df, sc):
    clustered_ = pd.DataFrame(index=df.index)
    clustered_['cluster'] = sc.labels_
    return clustered_

def contingency_counts(seq, random = True):
    result = probability_sensor_trigger(seq) if random else  probability_sensor_trigger(seq, random = False)
    prob_events = pd.DataFrame(columns=['cluster_0'])
    for event in events:
        if event not in result.keys():
            result[event] = 0
        new =result[event] # percentage
        probs = pd.concat([prob_events, pd.DataFrame(new, columns=['cluster_0'], index = [f'{event}'])])
        prob_events = probs
    return probs

def contingency_prob(seq, random = True):
    result = probability_sensor_trigger(seq) if random else  probability_sensor_trigger(seq, random = False)
    prob_events = pd.DataFrame(columns=['cluster_0'])
    for event in events:
        if event not in result.keys():
            result[event] = 0
        new = np.array([round(result[event]/sum(result.values())*100, 2)]) # in percentage
        probs = pd.concat([prob_events, pd.DataFrame(new, columns=['cluster_0'], index = [f'{event}'])])
        prob_events = probs
    return probs

def get_array(seq, random=True):
    result = probability_sensor_trigger(seq) if random else  probability_sensor_trigger(seq, random = False)
    t = []
    for event in events:
        if event not in result.keys():
            result[event] = 0
        t.append(result[event])
    return t

def get_array_percent(seq, random=True):
    result = probability_sensor_trigger(seq) if random else probability_sensor_trigger(seq, random=False)
    t = []
    for event in events:
        if event not in result.keys():
            result[event] = 0
        t.append(round(result[event]/sum(result.values())*100,2))
    return t

def countInRange(cluster, x, y):
    # initialize result
    cluster = cluster.index.to_numpy()
    count = 0
    n = len(cluster)
    for i in range(n):
        # check if element is in range
        if (cluster[i] >= x and cluster[i] <= y):
            count += 1
    total = round(count / len(cluster) *100, 3)
    print(f"{total}% of cluster located in time period {x}:{y} ")
    # e.g., in cluster_x, how many entries are within the interval 660:980
    return total