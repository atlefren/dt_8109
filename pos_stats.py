import pandas as pd

df = pd.read_csv('./data_v2/products_in_pos_purchases.csv')

print len(df['product_id'].unique())
