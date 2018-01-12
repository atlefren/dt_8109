import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from get_events import get_dataframe

# create the 10% subset for manual categorization
# df = get_dataframe()
# subset = df.sample(frac=0.1)
# subset[['event_id', 'max_capacity', 'start', 'end', 'name', 'event_created']].to_csv('./events_to_categorize.csv', index=False)

# read the categorized subset, shuffle it
df = pd.read_csv(
    # 'data_v2/events_sample2.csv',
    './events_to_categorize.csv',
    sep=';',
    parse_dates=['start', 'end', 'event_created'],
    date_parser=pd.core.tools.datetimes.to_datetime
).sample(frac=1).reset_index(drop=True)


# split in train set and holdout set (equal size)
split = np.array_split(df, 2)
train_set = split[0]
holdout_set = split[1]


# split in train set and holdout set (90/10)
'''
split = np.array_split(df, 10)
train_set = pd.concat((split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7], split[8]))
holdout_set = split[9]
'''

# tokenize
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_set['name'])

# compute tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# train the classifier
clf = MultinomialNB().fit(X_train_tfidf, train_set['category'])


# run trough the holdout set
# and predict the category
# count correct categories

num_corrent = 0
for row in holdout_set.itertuples():
    name = getattr(row, 'name')
    actual_cat = getattr(row, 'category')
    X_new_counts = count_vect.transform([name])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted_cat = clf.predict(X_new_tfidf)[0]
    if actual_cat == predicted_cat:
        num_corrent += 1

print (float(num_corrent) / float(len(holdout_set.index))) * 100.0

# classify all instances
'''
df_complete = pd.read_csv(
    # 'data_v2/events_sample2.csv',
    './data_v2/events.csv',
    sep=',',
    parse_dates=['start', 'end', 'event_created'],
    date_parser=pd.core.tools.datetimes.to_datetime
)

# optionally remove the manually categorized events
# df_complete = df_complete[~df_complete['event_id'].isin(train_set['event_id'].tolist())]


ids = []
categories = []
names = []
for row in df_complete.itertuples():
    name = getattr(row, 'name')
    X_new_counts = count_vect.transform([name])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted_cat = clf.predict(X_new_tfidf)[0]
    ids.append(getattr(row, 'event_id'))
    categories.append(predicted_cat)
    names.append(name)

categorized = pd.DataFrame({
    'name': names,
    'event_id': ids,
    'category': categories
})


categorized[['event_id', 'category']].to_csv('./events_categorized_nb.csv', index=False)

print categorized[categorized['category'] == 'concert'].head()

#print categorized.groupby(['category']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
'''
