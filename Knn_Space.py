from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def fit_knn(data, y):
    new_x = data[["X Coordinate", "Y Coordinate"]]
    kn = KNeighborsClassifier(50)
    return kn.fit(new_x, y)


