from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd


def fix(data):
    y = data["Primary Type"]
    X = data[["day_of_week", "day", "month", "Time", "District"]]
    pd.get_dummies(X, "District")
    X = X.drop("District", axis=1)
    # another_X = X["Time"].h

    X['Time'] = X['Time'].apply(lambda x: x.hour)

    another_X = X[["Time", "day_of_week"]]
    pd.get_dummies(X, "Time")
    # another_X.drop("Time", axis=1)
    fit(another_X, y)


def fit(X, y):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)
    y_hat = clf.predict(X)
    print("Accuracy:", metrics.accuracy_score(y, y_hat))

