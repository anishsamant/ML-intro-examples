import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
df=pd.read_csv('C:/Users/ANISH/Python_Demo/hepatitis1.csv',header=None,na_values=['?'])
print(df)

li=df.columns.tolist()
nanlist=df.columns[df.isnull().any()].tolist()

print(li)
print(nanlist)
print()

'''df_mean=df
for i in range(len(nanlist)):
    df_mean[nanlist[i]]=df[nanlist[i]].fillna(df[nanlist[i]].mean())

print(df_mean)'''

'''df_dropped=df.dropna();
print('Original size is:- ',df.shape)
print('Size after dropping',df_dropped.shape)

df_zero=df.fillna(0)
print(df_zero)'''

df[0].fillna(df.groupby(2)[0].transform('mean'),inplace=True)
df[1].fillna(df.groupby(2)[1].transform('mean'),inplace=True)
df[2].fillna(df.groupby(2)[2].transform('mean'),inplace=True)
df[3].fillna(df.groupby(2)[3].transform('mean'),inplace=True)
df[4].fillna(df.groupby(2)[4].transform('mean'),inplace=True)
df[5].fillna(df.groupby(2)[5].transform('mean'),inplace=True)
df[6].fillna(df.groupby(2)[6].transform('mean'),inplace=True)
df[7].fillna(df.groupby(2)[7].transform('mean'),inplace=True)
df[8].fillna(df.groupby(2)[8].transform('mean'),inplace=True)
df[9].fillna(df.groupby(2)[9].transform('mean'),inplace=True)
df[10].fillna(df.groupby(2)[10].transform('mean'),inplace=True)
df[11].fillna(df.groupby(2)[11].transform('mean'),inplace=True)
df[12].fillna(df.groupby(2)[12].transform('mean'),inplace=True)
df[13].fillna(df.groupby(2)[13].transform('mean'),inplace=True)
df[14].fillna(df.groupby(2)[14].transform('mean'),inplace=True)
df[15].fillna(df.groupby(2)[15].transform('mean'),inplace=True)
df[16].fillna(df.groupby(2)[16].transform('mean'),inplace=True)
df[17].fillna(df.groupby(2)[17].transform('mean'),inplace=True)
df[18].fillna(df.groupby(2)[18].transform('mean'),inplace=True)
df[19].fillna(df.groupby(2)[19].transform('mean'),inplace=True)

print(df)

df.hist()                                                              #histogram
df.plot(kind='density',subplots=True,layout=(4,5), sharex=False)       #curve graph
scatter_matrix(df)                                                     #scatter matrix
plt.show()
print(df.corr())                                        #shows co-relation between each pir of attributes
print(df.describe())
df.to_csv('C:/Users/ANISH/Python_Demo/modified.csv')
