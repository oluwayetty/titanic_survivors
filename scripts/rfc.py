import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as acc

# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
#print first 5 rows
# print(titanic.head(5))
# print(titanic.describe())

            # """Cleaning up our training data sets"""
# The titanic variable is available here.
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#print(titanic.describe())

# Find all the unique genders -- the column appears to contain only male and female.
# print(titanic["Sex"].unique())

# Replace all the occurences of male with the number 0.
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# Find all the unique values for "Embarked".
# print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

                    # """Cleaning up our test data sets"""

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

                        # """Let's build our model"""

# splitting my arrays in ratio of 30:70 percent
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(titanic[predictors], titanic['Survived'], test_size=0.4, random_state=0)
print titanic.shape
# Initialize the algorithm class
alg = RFC(n_estimators=30, random_state=0).fit(features_train, labels_train)
prob_predictions_class_test = alg.predict(features_test)

accuracy = acc(labels_test, prob_predictions_class_test, normalize=True,sample_weight=None)
print 'accuracy', accuracy
