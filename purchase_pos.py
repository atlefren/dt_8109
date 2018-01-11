import pandas as pd
import numpy
import datetime

print 'Stats for purchases'

df = pd.read_csv('data_v2/ticket_purchases.csv')

print 'Num purchases: %s' % len(df.index)


print df.fillna(-1).groupby(['latitude', 'longitude']).size().reset_index(name='counts').sort_values(['counts'], ascending=False)