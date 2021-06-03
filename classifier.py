import pandas as pd
import numpy as np
import fit

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
    data[["day", "month", "year_and_time"]] = data["Date"].str.split("/", expand=True)
    data["Time"] = pd.to_datetime(data["Time"]).dt.time
    data["Updated On"] = pd.to_datetime(data["Updated On"], dayfirst=True)
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)
    data["days_update"] = (data["Updated On"] - data["Date"]) / pd.offsets.Day(1)
    data = data.drop("new_date", axis=1)
    data = data.drop("year_and_time", axis=1)
    data = data.drop("Date", axis=1)
    data = data.drop("Updated On", axis=1)
    data['Arrest'] = data['Arrest'].apply({True: 1, False: 0}.get)
    data['Domestic'] = data['Domestic'].apply({True: 1, False: 0}.get)

    data = pd.get_dummies(data, columns=["Location Description", "Community Area"])
    data['Time'] = data['Time'].apply(lambda x: x.hour)

    return data, Y


def predict(X):
    pass


def send_police_cars(X):
    pass


X, y = load_data("train_data.csv")
fit.fix(X, y)

