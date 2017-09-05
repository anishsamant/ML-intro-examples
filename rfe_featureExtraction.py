import pandas as pd
from sklearn.feature_selection import RFE     #Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
df=pd.read_csv('C:/Users/ANISH/Python_Demo/hepatitis1.csv',header=None,na_values=['?'])
print(df)

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

values=df.values
X=values[:,0:df.shape[1]-1]         #contains values from column 0 to 18
Y=values[:,df.shape[1]-1]           #contains valies from last column
#print(X)
#print(Y)

model=LogisticRegression()
rfe=RFE(model,5)
fit=rfe.fit(X,Y)

print('Number of features:- ',fit.n_features_)
print('Selected features:- ',fit.support_)
print('Feature Ranking:- ',fit.ranking_)


df.to_csv('C:/Users/ANISH/Python_Demo/modified.csv')
