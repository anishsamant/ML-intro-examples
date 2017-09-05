from sklearn import datasets
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd


iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GaussianNB()

df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):

            eclf = VotingClassifier(estimators=[('Logistic Regression', clf1), ('Random Forest', clf2), ('Naive Bayes', clf3)], weights=[0.3,1,1], voting="hard")
            scores = cross_validation.cross_val_score(eclf, X, y, cv=5, scoring='accuracy')
            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            i += 1

df1 = df.sort_values(by=['mean'], ascending=False)
print(df1)