import os.path

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from get_events import get_time_of_year


pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)


def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")


def check_season(row, season):
    if get_time_of_year(row) == season:
        return 1
    return 0


def add_categories(df):
    cat = pd.read_csv('./events_categorized_nb.csv')
    categories = cat['category'].unique()

    def category_for_event(row):

        return cat.loc[cat['event_id'] == row['event_id']]['category'].values[0]

    def check_category(row, category):
        if category_for_event(row) == category:
            return 1
        return 0

    for category in categories:
        df['is_%s' % category] = df.apply(lambda row: check_category(row, category), axis=1)

    return df


def get_price(row, df_tickets):
    tickets = df_tickets.loc[df_tickets['TICKET_FOR_EVENT_ID'] == row['event_id']]
    return tickets['TICKET_PRICE'].min() / 100.0


def get_sold_tickets(filename):

    if os.path.isfile(filename):
        return pd.read_csv(
            filename,
            parse_dates=['start', 'end', 'event_created'],
            date_parser=pd.core.tools.datetimes.to_datetime
        )

    df = pd.read_csv(
        'data_v2/events.csv',
        parse_dates=['start', 'end', 'event_created'],
        date_parser=pd.core.tools.datetimes.to_datetime
    )
    df = df[~df.name.str.contains('test', case=False)].copy()

    # load supporting files
    tickets = pd.read_csv('data_v2/tickets.csv')
    purchases = pd.read_csv('data_v2/ticket_purchases.csv')

    def num_tickets_sold(row, tickets, purchases):
        event_id = row['event_id']
        tickets_for_event = tickets.loc[tickets['TICKET_FOR_EVENT_ID'] == event_id]['PURCHASE_ID']
        return len(tickets_for_event)

    df['tickets_sold'] = df.apply(lambda row: num_tickets_sold(row, tickets, purchases), axis=1, reduce=True)

    df_sold = df[df['tickets_sold'] > 0]
    # df_sold_out = df[df['tickets_sold'] >= df['max_capacity']]
    #df_sold_90 = df[df['tickets_sold'] >= df['max_capacity'] * 0.9]
    df_sold.to_csv(filename, index=False)
    return df_sold


def print_accuracy(df, percentage):
    num_corrent = 0
    for row in df.itertuples():
        variables = []
        for feature in features:
            variables.append(getattr(row, feature))
        predicted = dt.predict([variables])[0]
        actual = getattr(row, 'met_sales_goal')
        if actual == predicted:
            num_corrent += 1

    num_met = df[df['tickets_sold'] >= df['max_capacity'] * percentage]
    print 'num events: %s' % len(df.index)
    print 'num met %s sold: %s' % (percentage, len(num_met.index))
    print 'accuracy: %s' % ((float(num_corrent) / float(len(df.index))) * 100.0)

percentage = 0.9

df = get_sold_tickets('./events_sold_tickets.csv')
df['duration'] = df.apply(lambda row: (row['end'] - row['start']).total_seconds(), axis=1)

tickets = pd.read_csv('data_v2/tickets.csv')
df['price'] = df.apply(lambda row: get_price(row, tickets), axis=1)

for season in ['winter', 'spring', 'summer', 'fall']:
    df['is_%s' % season] = df.apply(lambda row: check_season(row, season), axis=1)


df['met_sales_goal'] = df.apply(lambda row: 1 if row['tickets_sold'] >= row['max_capacity'] * percentage else 0, axis=1)

df = add_categories(df)

split = np.array_split(df, 2)
train_set = split[0]
holdout_set = split[1]

features = ['duration', 'price', 'is_winter', 'is_spring', 'is_summer', 'is_fall', 'is_concert',  'is_stage',  'is_course', 'is_fooddrinks', 'is_exhibition',  'is_unknown',  'is_festival',  'is_other']

X = df[features]
y = df['met_sales_goal']
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)

print_accuracy(train_set, percentage)
