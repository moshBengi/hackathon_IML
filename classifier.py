import pandas as pd
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import fit
import pickle

crimes_dict = {'BATTERY': 0, 'THEFT' : 1, 'CRIMINAL DAMAGE' : 2, 'DECEPTIVE PRACTICE' : 3, 'ASSAULT' : 4}


def load_data_1(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    data = pd.read_csv(path)
    data = data.fillna(0)
    data.drop(data.columns[[0]], axis=1, inplace=True)
    data = data.drop(
        ["IUCR", "FBI Code", "Description", "ID", "Case Number", "Year", "Latitude", "Longitude", "Location",
          "Block", "Beat", "District", "Ward"], axis=1)
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
    data['month'] = data['month'].apply(lambda x: (np.cos(float(x) * math.pi / 12)))
    data['day'] = data['day'].apply(lambda x: (np.cos(float(x) * math.pi / 31)))
    data['Time'] = data['Time'].apply(lambda x: (np.cos(float(x) * math.pi / 24)))
    data['day_of_week'] = data['day_of_week'].apply(lambda x: (np.cos(float(x) * math.pi / 7)))

    return data


def load_data(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    data = pd.read_csv(path)
    data = data.fillna(0)
    Y = np.array(data["Primary Type"])
    data.drop(data.columns[[0]], axis=1, inplace=True)
    data = data.drop(
        ["IUCR", "FBI Code", "Description", "ID", "Case Number", "Year", "Latitude", "Longitude", "Location",
         "Block", "Beat", "District", "Ward"], axis=1)
    data[["new_date", "Time"]] = data["Date"].str.split(" ", 1, expand=True)
    data["new_date"] = pd.to_datetime(data["new_date"], dayfirst=True)
    data["day_of_week"] = data["new_date"].dt.dayofweek
    data["day_of_week"] = (data["day_of_week"] + 2) % 7
    data[["month", "day", "year_and_time"]] = data["Date"].str.split("/", 2, expand=True)
    data["Time"] = pd.to_datetime(data["Time"]).dt.time
    data["Updated On"] = pd.to_datetime(data["Updated On"], dayfirst=True)
    data["Date2"] = pd.to_datetime(data["Date"], dayfirst=True)
    data["days_update"] = (data["Updated On"] - data["Date2"]) / pd.offsets.Day(1)
    data = data.drop("new_date", axis=1)
    data = data.drop("Updated On", axis=1)
    data = data.drop("year_and_time", axis=1)
    data['Arrest'] = data['Arrest'].apply({True: 1, False: 0}.get)
    data['Domestic'] = data['Domestic'].apply({True: 1, False: 0}.get)
    data = pd.get_dummies(data, columns=["Location Description", "Community Area"])
    data['Time'] = data['Time'].apply(lambda x: x.hour)
    data['month'] = data['month'].apply(lambda x: (np.cos(float(x) * math.pi / 12)))
    data['day'] = data['day'].apply(lambda x: (np.cos(float(x) * math.pi / 31)))
    data['Time'] = data['Time'].apply(lambda x: (np.cos(float(x) * math.pi / 24)))
    data['day_of_week'] = data['day_of_week'].apply(lambda x: (np.cos(float(x) * math.pi / 7)))
    data = data.reset_index()
    return data, Y


def predict(X):
    test = load_data_1(X)
    X_train = pickle.load(open("columns.p", "rb"))
    print(type(X_train))
    print(type(X_train.columns))
    cls = pickle.load(open("weights.p", "rb"))
    X.train.head()
    missing_cols = set(X_train.columns) - set(test.columns)
    for c in missing_cols:
        test[c] = 0
    test = test[X_train.columns]
    test = test.drop("X Coordinate", axis=1)
    test = test.drop("Y Coordinate", axis=1)
    y_hat = cls.predict(test)
    return np.vectorize(crimes_dict.get)(y_hat)


def send_police_cars(X):
    length = len(X)
    length = (30 // length) + 1
    lst2 = []
    for i in range(X.shape[0]):
        x = X[i]
        month = x.split("/")[0]
        x = pd.to_datetime(x)
        d = {'day_of_week': [(x.dayofweek + 2) % 7], 'month': [month]}
        new_point = pd.DataFrame(data=d)
        new_point['day_of_week'] = new_point['day_of_week'].apply(lambda y: (np.cos(float(y) * math.pi / 7)))
        new_point['month'] = new_point['month'].apply(lambda y: (np.cos(float(y) * math.pi / 12)))

        data = pickle.load(open("data.p", "rb"))
        temp_data = data[["day_of_week", "month"]]
        nbrs = NearestNeighbors(n_neighbors=30, algorithm='ball_tree').fit(temp_data)
        distances, indices = nbrs.kneighbors(new_point)
        indices = indices.T
        arr = []
        for i in range(indices.shape[0]):
            arr.append(indices[i][0])
        df = pd.DataFrame(data, index=arr)
        df = df[["X Coordinate", "Y Coordinate", "Date"]]
        lst = df.to_numpy()
        for j in range(length):
            lst2.append(tuple(lst[j]))
    lst2 = lst2[:30]
    return lst2


# if __name__ == '__main__':
#     X, y = load_data_1("train_total.csv")
#
#     t=X.iloc[[0,1]]
#     pickle.dump(t, open("columns.p", "wb"))
#     k=pickle.load(open("columns.p", "rb"))
#     cl=fit.fit_random_forest(X,y)
#     pickle.dump(cl, open("weights.p", "wb"))
#     cls = pickle.load(open("weights.p", "rb"))
#
    # X_t, y_t = load_data_1("test_total.csv")
#
#     # missing_cols = set( X.columns ) - set( X_t.columns )
#     # for c in missing_cols:
#     #     X_t[c] = 0
#     # X_t = X_t[X.columns]
#     # X_t = X_t.drop("X Coordinate", axis=1)
#     # X_t = X_t.drop("Y Coordinate", axis=1)
#
#     # y_hat_4 = cls.predict(X_t)
#     # print("Accuracy:", metrics.accuracy_score(y_t, predict("test_total.csv")))
#
#     # data, y = load_data("total.csv")
#     # pickle.dump(data, open("data.p", "wb"))
#     # k=pickle.load(open("data.p", "rb"))
#     # r = 5
#     print(send_police_cars("01/04/2021  21:20:00"))




