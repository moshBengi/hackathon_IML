import pandas as pd
import numpy as np
import Knn_Space
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def load_data(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    data = pd.read_csv(path)
    data = data.dropna().drop_duplicates()
    Y = np.array(data["Primary Type"])
    data.drop(data.columns[[0, 1]], axis = 1, inplace = True)
    data = data.drop(
        ["IUCR", "FBI Code", "Description", "ID", "Case Number", "Year", "Latitude", "Longitude", "Location",
         "Primary Type", "Block", "Beat", "District", "Ward"], axis=1)
    data[["new_date", "Time"]] = data["Date"].str.split(" ", 1, expand=True)
    data["new_date"] = pd.to_datetime(data["new_date"], dayfirst=True)
    data["day_of_week"] = data["new_date"].dt.dayofweek
    data["day_of_week"] = (data["day_of_week"] + 2) % 7
    data[["month", "day", "year_and_time"]] = data["Date"].str.split("/", 2, expand=True)
    data["Time"] = pd.to_datetime(data["Time"]).dt.time
    data["Updated On"] = pd.to_datetime(data["Updated On"], dayfirst=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data["days_update"] = (data["Updated On"] - data["Date"]) / pd.offsets.Day(1)
    data = data.drop("new_date", axis=1)
    data = data.drop("Date", axis=1)
    data = data.drop("Updated On", axis=1)
    data = data.drop("year_and_time", axis=1)
    data['Arrest'] = data['Arrest'].apply({True: 1, False: 0}.get)
    data['Domestic'] = data['Domestic'].apply({True: 1, False: 0}.get)

    data = pd.get_dummies(data, columns=["Location Description", "Community Area"])
    data['Time'] = data['Time'].apply(lambda x: x.hour)


    return data, Y


def predict(X):
    pass


def send_police_cars(X):
    pass


if __name__ == '__main__':

    X, y = load_data("train_data.csv")
    new_x = X[["X Coordinate", "Y Coordinate"]]
    kn = Knn_Space.fit_knn(new_x, y)
    clf = fit.fit_random_forest(X, y)

    X_t, y_t = load_data("valid_data.csv")
    new_x_t = X_t[["X Coordinate", "Y Coordinate"]]
    # X_t = X_t.drop("X Coordinate", axis=1)
    # X_t = X_t.drop("Y Coordinate", axis=1)

    y_hat = kn.predict(new_x_t)
    y_p = kn.predict_proba(new_x_t)
    print("Accuracy:", metrics.accuracy_score(y_t, y_hat))
    print("predict prob: ", y_p)
