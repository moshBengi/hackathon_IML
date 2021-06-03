from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier


def fit_random_forest(data, y):
    data = data.drop("X Coordinate", axis=1)
    data = data.drop("Y Coordinate", axis=1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data, y)
    return clf


def fit_knn_space(data, y):
    new_x = data[["X Coordinate", "Y Coordinate"]]
    kn = KNeighborsClassifier(50)
    return kn.fit(new_x, y)


def fit_knn_time(data, y):
    new_x = data[["month", "Time", "day_of_week"]]
    kn = KNeighborsClassifier(47)
    return kn.fit(new_x, y)


def fit_linear_regression(data, y):
    data = data.drop("X Coordinate", axis=1)
    data = data.drop("Y Coordinate", axis=1)
    cls = OneVsOneClassifier(LogisticRegression(solver='liblinear').fit(data, y))
    cls.fit(data, y)
    return cls

