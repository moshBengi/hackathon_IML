import pandas as pd
import numpy as np

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}


def load_data(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    x_train = pd.read_csv(path)
    # x_train = data.sample(frac=0.43)
    # x_temp = data.drop(x_train.index)
    # x_valid = x_temp.sample(frac=0.50)
    # x_test = x_temp.drop(x_valid.index)
    # x_train.to_csv("train_data.csv")
    # x_valid.to_csv("valid_data.csv")
    # x_test.to_csv("test_data.csv")

    x_train = x_train.drop(["IUCR", "FBI Code", "Description"], axis=1)
    # data[["day", "month", "year"]] = data["date"].str.split("/", expand = True)


    return x_train


def predict(X):
    pass


def send_police_cars(X):
    pass


load_data("train_data.csv")
