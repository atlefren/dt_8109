import pandas as pd

tickets = pd.read_csv('data_v2/tickets.csv')
purchases = pd.read_csv('data_v2/ticket_purchases.csv')


def users_for_event(event_id):
    print event_id
    purchase_ids = tickets.loc[tickets['TICKET_FOR_EVENT_ID'] == event_id]['PURCHASE_ID']
    return purchases.loc[purchases['purchase_id'].isin(purchase_ids)]['user_id'].unique()

event_id = 2939690836


print users_for_event(event_id)
