import numpy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os.path
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


from dist import season_dist, capacity_dist, duration_dist, users_dist
from get_events import get_dataframe

DO_PICKLE = False

# get the dataframe
df = get_dataframe()

# create a tf-idf vectorizer for comparing names of the events
# then build a tf-idf-matrix and compute cosine distances for all names
vectorizer = TfidfVectorizer(analyzer='word', stop_words=None)
tfidf_matrix = vectorizer.fit_transform(df['name'])
distances = cosine_distances(tfidf_matrix, tfidf_matrix)


# the distance between two names is then just a lookup in the distance matrix
def name_dist(d1, d2):
    return distances[d1['idx']][d2['idx']]


# utility for extracting data from what is now a numpy array derived from a pandas row
# (or something)
def get_data(e):
    return {
        'idx': e[9],
        'event_id': e[0],
        'max_capacity': e[1],
        'name': e[4],
        'duration': e[6],
        'time_of_year': e[7],
        'users': e[8].split(',')
    }

# constants used for duration distance
max_duration = df['duration'].max().total_seconds()
min_duration = 0


# custom function for calculating distance between two features
def event_dist(e1, e2):
    d1 = get_data(e1)
    d2 = get_data(e2)

    # get the induvidual distances in <0, 1>-range, optionally scale them
    name = name_dist(d1, d2)
    cap = capacity_dist(d1['max_capacity'], d2['max_capacity'])
    dur = duration_dist(d1['duration'], d2['duration'], min_duration, max_duration)
    season = season_dist(d1['time_of_year'], d2['time_of_year'])
    users = users_dist(d1['users'], d2['users'])
    return name + cap + dur + season + users


# get the linkage matrix used by the hirarcical clustering
def get_linkage(df):
    if DO_PICKLE:
        pickle_name = './events_linkage.pickle'
        if os.path.isfile(pickle_name):
            return numpy.load(pickle_name)
    data_dist = pdist(df, event_dist)
    data_link = linkage(data_dist, 'single')
    if DO_PICKLE:
        data_link.dump(pickle_name)
    return data_link


data_link = get_linkage(df)


threshold = 1.4

# create a dendrogram (fails on complete dataset)
#dendrogram(data_link, color_threshold=threshold)
#plt.savefig("scipy-dendrogram.png")

# cluster the data
clusters = fcluster(data_link, threshold, criterion='distance')

# my "clever" way of working around pandas
cluster_mapping = {}
for idx, cluster_id in enumerate(clusters):
    event_id = df.iloc[idx]['event_id']
    cluster_mapping[event_id] = cluster_id


def assign_cluster(row):
    return cluster_mapping.get(row['event_id'], None)

df['cluster'] = df.apply(assign_cluster, axis=1)

print df[['cluster', 'event_id']].sort_values(by=['cluster'])

print df.groupby(['cluster']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)

# write event-id and cluster to csv
df[['event_id', 'cluster']].sort_values(by=['cluster']).to_csv('./events_cat.csv', index=False)
