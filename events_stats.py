import pandas as pd
import numpy
import datetime

print 'Stats for events'

df_events = pd.read_csv('data_v2/events.csv', parse_dates=['start', 'end', 'event_created'], date_parser=pd.core.tools.datetimes.to_datetime)

print 'Num events: %s' % len(df_events.index)

num_test = numpy.asscalar(df_events[df_events.name.str.contains('test', case=False)]['name'].count())

print 'contains test: %s' % num_test

df_events_notest = df_events[~df_events.name.str.contains('test', case=False)].copy()

# df_events_notest['start'] = pd.to_datetime(df_events_notest['start'])
# df_events_notest['end'] = pd.to_datetime(df_events_notest['end'])
# df_events_notest['event_created'] = pd.to_datetime(df_events_notest['event_created'])

# print df_events_notest.head()

# print df_events_notest['event_created'].min(), df_events_notest['event_created'].max()

# first = pd.Timestamp(datetime.date(2010, 1, 1))
# last = pd.Timestamp(datetime.date(2018, 1, 1))

# print df_events_notest[(df_events_notest['start'] < first) | (df_events_notest['start'] > last)]

# print df_events_notest[(df_events_notest['end'] < first) | (df_events_notest['end'] > last)]

# set the duration of the event
#df_events_notest['duration'] = df_events_notest.apply(lambda row: row['end'] - row['start'], axis=1)

#print df_events_notest.head()


tickets = pd.read_csv('data_v2/tickets.csv')
purchases = pd.read_csv('data_v2/ticket_purchases.csv')


def tickets_for_event(row):
    event_id = row['event_id']
    purchase_ids = tickets.loc[tickets['TICKET_FOR_EVENT_ID'] == event_id]['PURCHASE_ID']
    #return ','.join(str(v) for v in user_ids.tolist())
    return len(purchase_ids.tolist())

df_events_notest['num_tickets'] = df_events_notest.apply(lambda row: tickets_for_event(row), axis=1, reduce=True)

print df_events_notest.fillna('undefined').groupby(['num_tickets']).size().reset_index(name='count')
