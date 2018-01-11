import pandas as pd


print 'Stats for purchases'

df = pd.read_csv('data_v2/ticket_purchases.csv')

print 'Num purchases: %s' % len(df.index)

print 'unique users: %s' % df['user_id'].nunique()

print df.fillna(-1).groupby(['user_id']).size().reset_index(name='num_bought').groupby('num_bought').size().reset_index(name='count')

print df.fillna(-1).groupby(['latitude', 'longitude']).size().reset_index(name='counts').sort_values(['counts'], ascending=False)

print df['checkout_method'].unique()
print df.fillna('undefined').groupby(['checkout_method']).size().reset_index(name='count')

#.sort_values(['num_bought'], ascending=False)