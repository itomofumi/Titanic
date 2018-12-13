



import numpy as np
import pandas as pd
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("test.csv", dtype={"Age": np.float64}, )
train.head(10)

train_corr = train.corr()
train_corr


def correct_data(train_data, test_data):

    # Make missing values ​​for training data from test data as well
    train_data.Age = train_data.Age.fillna(test_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(test_data.Fare.median())

    test_data.Age = test_data.Age.fillna(test_data.Age.median())
    test_data.Fare = test_data.Fare.fillna(test_data.Fare.median())    

    train_data = correct_data_common(train_data)
    test_data = correct_data_common(test_data)    

    return train_data,  test_data

def correct_data_common(titanic_data):
    titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S")
    titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])

    return titanic_data

train_data,  test_data = correct_data(train, test)

train_corr = train.corr()
train_corr

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
​
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
​
parameters = {
        'n_estimators'      : [5, 10, 20, 30],
        'max_depth'         : [3, 5, 8],
        'random_state'      : [0],
}
​
predictors = ["Pclass", "Sex", "Age", "Parch", "Fare", "Embarked"]
​
models = []
​
models.append(("GradientBoosting", GradientBoostingClassifier(n_estimators=26)))
models.append(("LogisticRegression",LogisticRegression()))
models.append(("SVC",SVC()))
models.append(("LinearSVC",LinearSVC()))
models.append(("KNeighbors",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("RandomForest",GridSearchCV(RandomForestClassifier(), parameters)))
models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))


results = []
names = []
for name,model in models:
    result = cross_val_score(model, train_data[predictors], train_data["Survived"],  cv=3)
    names.append(name)
    results.append(result)
​
​
for i in range(len(names)):
    print(names[i],results[i].mean())

##------------------------------

alg = RandomForestClassifier()
alg.fit(train_data[predictors], train_data["Survived"])

predictions = alg.predict(test_data[predictors])

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('submission.csv', index=False)

