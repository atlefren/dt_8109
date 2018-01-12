import pandas as pd
import numpy as np

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

df = pd.read_csv(
    # 'data_v2/events_sample2.csv',
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

# df_sold = df[df['tickets_sold'] > 0]
df_sold_90 = df[df['tickets_sold'] >= df['max_capacity'] * 0.9]
# df_sold_out = df[df['tickets_sold'] >= df['max_capacity']]

split = np.array_split(df_sold_90, 2)
train_set = split[0]
holdout_set = split[1]

print train_set
