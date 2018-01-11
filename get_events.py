import os.path

import pandas as pd


DO_PICKLE = False


def get_time_of_year(row):
    month = row['start'].to_pydatetime().month
    if 3 <= month > 6:  # Spring runs from March 1 to May 31;
        return 'spring'
    elif 6 <= month < 9:  # Summer runs from June 1 to August 31;
        return 'summer'
    elif 9 <= month < 12:  # Fall (autumn) runs from September 1 to November 30; and
        return 'fall'
    else:  # Winter runs from December 1 to February 28 (February 29 in a leap year).
        return 'winter'


def users_for_event(row, tickets, purchases):
    event_id = row['event_id']
    purchase_ids = tickets.loc[tickets['TICKET_FOR_EVENT_ID'] == event_id]['PURCHASE_ID']
    user_ids = purchases.loc[purchases['purchase_id'].isin(purchase_ids)]['user_id'].unique()
    # pandas goes bananas when i return a Series or a list...
    return ','.join(str(v) for v in user_ids.tolist())


def get_idx(row, df):
    base = df.index.get_indexer_for((df[df.event_id == row['event_id']].index))
    return base[0]


def get_dataframe():
    if DO_PICKLE:
        pickle_name = './events.pickle'
        if os.path.isfile(pickle_name):
            return pd.read_pickle(pickle_name)

    # read in the csv file, handle dates
    df = pd.read_csv(
        # 'data_v2/events_sample2.csv',
        'data_v2/events.csv',
        parse_dates=['start', 'end', 'event_created'],
        date_parser=pd.core.tools.datetimes.to_datetime
    )

    # remove events containing thr word "test"
    df = df[~df.name.str.contains('test', case=False)].copy()

    # compute duration
    df['duration'] = df.apply(lambda row: row['end'] - row['start'], axis=1)

    # load supporting files
    tickets = pd.read_csv('data_v2/tickets.csv')
    purchases = pd.read_csv('data_v2/ticket_purchases.csv')

    # convert date to season
    df['time_of_year'] = df.apply(get_time_of_year, axis=1)

    # extract users for an event
    df['users'] = df.apply(lambda row: users_for_event(row, tickets, purchases), axis=1, reduce=True)

    # either am I stupid, else there is something rotten in the world of pandas - scipy interop
    # I neeeeed the effing row number [not some strange pandas index] ffs!
    df['idx'] = df.apply(lambda row: get_idx(row, df), axis=1, reduce=True)

    if DO_PICKLE:
        df.to_pickle(pickle_name)
    return df
