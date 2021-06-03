import numpy as np
from tqdm import tqdm


def council(forest, knn, X):
    forest_p = forest.predict_proba(X)
    knn_p = knn.predict_proba(X)

    p_tot = forest_p + knn_p
    y_hat = [np.argmax[p_tot[i]] for i in range(len(p_tot))]
    # for i in
    return y_hat