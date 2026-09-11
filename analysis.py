import pandas as pd
import numpy as np
from plotnine.data import diamonds

df = diamonds.copy()
duplicate_count = df.duplicated().sum()
zero_dimension_count = (df[['x','y','z']] == 0).any(axis=1).sum()

df = df.drop_duplicates().copy()
df['Dimension_Quality'] = np.where((df[['x','y','z']] > 0).all(axis=1), 'Valid', 'Invalid/Zero Dimension')
df = df[df['Dimension_Quality'] == 'Valid'].copy()

df['Volume_mm3'] = df.x * df.y * df.z
df['Price_per_Carat'] = df.price / df.carat
df['Size_Category'] = pd.cut(df.carat, [0,.5,1,1.5,2,np.inf],
                             labels=['<0.5 ct','0.5–1 ct','1–1.5 ct','1.5–2 ct','2+ ct'],
                             include_lowest=True)
df['Price_Tier'] = pd.qcut(df.price, 4, labels=['Budget','Mid','Premium','Luxury'])

cut_summary = df.groupby('cut', observed=True).agg(
    Records=('price','size'),
    Avg_Price=('price','mean'),
    Median_Price=('price','median'),
    Avg_Price_per_Carat=('Price_per_Carat','mean')
).sort_values('Avg_Price', ascending=False)

size_summary = df.groupby('Size_Category', observed=True).agg(
    Records=('price','size'), Revenue_Proxy=('price','sum'), Avg_Price=('price','mean')
)
size_summary['Record_Share'] = size_summary.Records / len(df)
size_summary['Revenue_Share'] = size_summary.Revenue_Proxy / df.price.sum()

corr = df[['carat','price','depth','table','x','y','z','Volume_mm3','Price_per_Carat']].corr()['price'].sort_values(ascending=False)

print('Raw rows:', 53940)
print('Exact duplicates removed:', duplicate_count)
print('Zero-dimension records removed:', zero_dimension_count)
print('Processed rows:', len(df))
print(cut_summary)
print(size_summary)
print(corr)
