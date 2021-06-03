
import pandas as pd
import numpy as np
from sklearn import model_selection
crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}




def load_data(path):
    """
    :param path: The path to load from the data.
    :return: After the preprocessing - The design matrix X, and the responses y.
    """
    data = pd.read_csv(path)
    data = data.drop(["ICUR", "FBI code", "description"], axis=1)
    x_train = data.sample(frac=0.43)
    x_temp = data.drop(x_train.index)
    x_valid = x_temp.sample(frac=0.50)
    x_test = x_temp.drop(x_valid.index)



    return data

def predict(X):
    pass

def send_police_cars(X):
    pass

load_data("Dataset_crimes.csv")