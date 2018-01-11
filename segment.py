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


# split in train set and holdout set
split = np.array_split(df, 2)
train_set = split[0]
holdout_set = split[1]

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
    #print name, actual_cat, predicted_cat
    if actual_cat == predicted_cat:
        num_corrent += 1

print (float(num_corrent) / float(len(holdout_set.index))) * 100.0
