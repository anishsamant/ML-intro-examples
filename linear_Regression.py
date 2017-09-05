import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('C:/Users/ANISH/Python_Demo/ex2data1.csv',header=None,na_values=['?'])
#print(df)

data=df.values

X=data[:,0:len(data[0])-1]
Y=data[:,len(data[0])-1]
#print(X)
#print(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

for j in range(2,11):
    mean=0
    for k in range(0,5):
        quadratic_featurizer=PolynomialFeatures(degree=j)
        x_train_quadratic=quadratic_featurizer.fit_transform(x_train)
        x_test_quadratic=quadratic_featurizer.fit_transform(x_test)

        regressor=LinearRegression()
        regressor.fit(x_train_quadratic,y_train)

        #regressor.fit(x_train,y_train)


        cnt=0
        diff=0
        '''for i in x_test:
            pred=regressor.predict(i)
            actual=y_test[cnt][0]
            cnt+=1
            diff=diff+((actual-pred)*(actual-pred))'''

        for i in x_test_quadratic:
            pred = regressor.predict(i.reshape(1,-1))
            diff = diff + ((pred[0][0] - y_test[cnt][0]) * (pred[0][0]-y_test[cnt][0]))
            cnt += 1

        mean=mean+diff
    mean=mean/5
    print(mean)


#print('Differences:- ', diff)

'''
pred=regressor.predict(x_test)
print(pred)'''
