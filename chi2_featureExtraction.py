import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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

#seperately storing attributes and class label from df in X & Y resp.
values=df.values
X=values[:,0:df.shape[1]-1]         #contains values from column 0 to 18
Y=values[:,df.shape[1]-1]           #contains valies from last column
#print(X)
#print(Y)

test=SelectKBest(score_func=chi2,k=4)       #criteria for selection is saved in test
fit=test.fit(X,Y)                           #fit() identifies features from X & Y which follow the test criteria
features=fit.transform(X)                   #features identified in fit() are extracted from X using transform() in features

print(features)                             #extracted features are printed

df.to_csv('C:/Users/ANISH/Python_Demo/modified.csv')
