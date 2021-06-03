from sklearn.ensemble import RandomForestClassifier


def fit_random_forest(data, y):
    data = data.drop("X Coordinate", axis=1)
    data = data.drop("Y Coordinate", axis=1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(data, y)
    return clf


def fit_knn(data, y):
    new_x = data[["X Coordinate", "Y Coordinate"]]
    kn = KNeighborsClassifier(50)
    return kn.fit(new_x, y)
