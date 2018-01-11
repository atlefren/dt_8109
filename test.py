import pandas as pd
df = pd.read_csv('events_cat.csv')
df = df.groupby(['cluster']).size().reset_index(name='count').sort_values(by=['count'], ascending=False)
#print df.columns.tolist()
print df.loc[df['count'] > 1].shape