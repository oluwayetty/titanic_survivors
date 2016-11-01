import pandas as pd

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the assigned variables.
titanic = pd.read_csv("../datasets/train.csv")
titanic_test = pd.read_csv("../datasets/test.csv")

#print first 5 rows
print(titanic.head(5))
print(titanic.describe())

# replace missing age columns wiht the mean of the total age of the passengers
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())

# Find all the unique genders -- the column appears to contain only male and female.
print(titanic["Sex"].unique())

# refactor the gender column into readable format for building our model.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

                    # cleaning up our test datasets

#replace missing age coulmns with the median of the age
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

#replace missing fare value of fare with the median of all fares
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# converting non-numeric values of sex to numeric
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

# replace missing values of embarked with "S"
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

#  converting non-numeric values of embarked to numeric
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

                    # On to machine learning(building our model)
from sklearn.linear_model import LogisticRegression as lg
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score as acc
# import numpy as np

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm class
alg = lg()
model = alg.fit(titanic[predictors], titanic["Survived"])

train_predictors = titanic[predictors]

# The target we're using to train the algorithm.
train_target = titanic["Survived"]

scores = cross_val_score(model, train_predictors, train_target, cv=10)
print scores
print scores.mean()

predictions = alg.predict(titanic_test[predictors])
print predictions

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

#writing our submission to csv.
submission.to_csv('../predictions.csv', sep=',', encoding='utf-8', index = False)
