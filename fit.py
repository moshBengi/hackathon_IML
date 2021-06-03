from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def fix(data, y):
    data = data.drop("X Coordinate", axis=1)
    data = data.drop("Y Coordinate", axis=1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data, y)
    return clf

