# predicting titanic survivors.

[Titanic](https://en.wikipedia.org/wiki/RMS_Titanic) RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning of 15 April 1912, after colliding with an iceberg during her maiden voyage from Southampton to New York City. Of the 2,224 passengers and crew aboard, more than 1,500 died in the sinking, making it one of the deadliest commercial peacetime maritime disasters in modern history.

Below is a brief details about each column:

PassengerId -- A numerical id assigned to each passenger.
Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
Name -- the name of the passenger.
Sex -- The gender of the passenger -- male or female.
Age -- The age of the passenger. Fractional.
SibSp -- The number of siblings and spouses the passenger had on board.
Parch -- The number of parents and children the passenger had on board.
Ticket -- The ticket number of the passenger.
Fare -- How much the passenger paid for the ticker.
Cabin -- Which cabin the passenger was in.
Embarked -- Where the passenger boarded the Titanic.

```datasets folder contains the training sets and test sets```

``` scripts folder contains the two different classifiers(LogisticRegression and RandomForestClassifier) used in making predictions to these datasets```

```predictions.csv is the predictions made from the datasets in a CSV format with two columns: PassengerId and Survived column. The survived column is the chance of survivor of each passenger on the ship. 1 means the passenger survived while 0 means otherwise :(```

I built a model for the datasets using two different classifiers from scikit-learn. Read the blog post [here](no link yet)
