import pandas as pd
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def load_data(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    data = pd.read_csv(path)
    data = data.drop(["IUCR", "FBI Code", "Description"], axis=1)
    data[["new_date", "Time"]] = data["Date"].str.split(" ", 1, expand = True)
    data["new_date"] = pd.to_datetime(data["new_date"], dayfirst=True)
    data["day_of_week"] = data["new_date"].dt.dayofweek
    data[["day", "month", "year_and_time"]] = data["Date"].str.split("/", expand = True)
    data["Time"] = pd.to_datetime(data["Time"]).dt.time

    data = data.drop("new_date", axis=1)
    data = data.drop("year_and_time", axis=1)

    # data = data.drop("year_and_time")
    # x_train = data.sample(frac=0.43)
    # x_temp = data.drop(x_train.index)
    # x_valid = x_temp.sample(frac=0.50)
    # x_test = x_temp.drop(x_valid.index)



    return data


def predict(X):
    pass


def send_police_cars(X):
    pass


load_data("train_data.csv")
