import numpy as np
import pylab as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from  wordcloud import WordCloud
import matplotlib.colors as mcolors

'''
TOP WORDS PER TOPIC
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

def wordcloud_freq(top_words):
    wordcloud = WordCloud(
            collocations=False,
            background_color='white', 
            max_words=5000,
            contour_width=3,
            colormap='winter',
            )
    wordcloud.generate_from_frequencies(dict(top_words))
    fig, ax = plt.subplots(figsize=[10,5])
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Plot the 5 top words 
def get_tf_idf_by_topic(df):
    tfidf_class = {}
    for t in df.topic.unique():
        topic=df.loc[df.topic == t]
        docs = list(topic.user_said)
        vectorizer = TfidfVectorizer(stop_words="english", min_df=0,)
        X = vectorizer.fit_transform(docs)
        tfidf_class[t] = [X,vectorizer]
    return tfidf_class

# Extract top words
def extract_top_n_words_per_topic(dict_, n=20):
    top_words = {}
    for t in dict_.keys():
        tfidf = dict_[t][1]
        vocab = tfidf.vocabulary_
        reverse_vocab = {v:k for k,v in vocab.items()}
        feature_names = tfidf.get_feature_names_out()
        tfidf_feature_names = np.array(feature_names)
        matrix=dict_[t][0]
        word_weight = np.array(matrix.sum(axis=0)/(matrix.shape[0]))
        importance = np.argsort(np.asarray(word_weight[0]).ravel())[::-1]
        top_n = [[tfidf_feature_names[i], word_weight[0][i]] for i in importance[:n]]
        top_words[t] = top_n
    return top_words

def plot_top_words(top_words, num):
    """
    Plot top 5 words for each extracted topic using TF-IDF scores
    It uses frequency of words to determine how relevant those words are to a given document
    """
    import matplotlib.colors as mcolors
    fig, axs = plt.subplots(ncols=2, nrows=5, figsize=(9,10), constrained_layout=True)
    colours = ['#191970', '#104E8B', '#4682B4', '#00B2EE', '#87CEFA', '#A2B5CD', '#FFA07A', '#EE6363', '#8B2252', '#DC143C']
    colours = [mcolors.to_rgba(c, alpha=0.85) for c in colours]
   
    y_pos = np.arange(num)
    for i in range(len(top_words)):
        labels = np.array(top_words[i])[:num,0]
        frequency = [float(number) for number in np.array(top_words[i])[:num,1]]
        axs[int(i/2), int(i%2)].barh(y_pos, frequency,align='center',color=colours[i])
        axs[int(i/2), int(i%2)].set_yticks(y_pos, labels=labels,fontsize='x-large')
        axs[int(i/2), int(i%2)].invert_yaxis()  
        axs[int(i/2), int(i%2)].set_title(f'Topic: {cluster_labels[i]}',fontsize='x-large')
        axs[int(i/2), int(i%2)].set_xlim(0,0.65)
        xticks=axs[int(i/2), int(i%2)].get_xticks()
        axs[int(i/2), int(i%2)].set_xticks(xticks,fontsize='x-large')
    plt.show()