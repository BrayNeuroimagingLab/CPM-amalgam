import scipy.stats as scistats
from sklearn.metrics import mean_squared_error
import numpy as np
import sklearn.linear_model as lm

def np_pearson_cor(x, y):
    """
    Return the pearson corr of two vectors using only numpy
    :param x: 1-d vector
    :param y: 1-d vector
    :return: float
    """
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)[0,0]

def partial_corr(x, y, Z):
    """
    Compute partial corr, controlling for random variables
    :param x: Random variable 1. shape=(n,)
    :param y: Random variable 2. shape=(n,)
    :param Z: controlling variables. shape=(n,p)
    :return: The partial correlation (float) of x and y
    """
    if Z is None: return np_pearson_cor(x, y)
    # Fit linear model for both vectors, then correlate their residuals
    slr1 = lm.LinearRegression().fit(Z, x)
    slr2 = lm.LinearRegression().fit(Z, y)
    pred1 = slr1.predict(Z)
    pred2 = slr2.predict(Z)
    return np_pearson_cor(x - pred1, y - pred2)

def error(y_pred, y_true, Z_test):
    """
    :param Z_test: Test confounds to regress out in partial corr
    :return: vector: Pearson accuracy, Spearman accuracy, partial corr
                     Root-Mean-Squared Error (RMSE), Normalized MSE

    """
    NMSE = np.mean((y_pred - y_true) ** 2) / np.mean((y_true - np.mean(y_true)) ** 2)
    return np_pearson_cor(y_pred, y_true), scistats.spearmanr(y_pred, y_true)[0], partial_corr(y_pred, y_true, Z_test), \
           np.sqrt(mean_squared_error(y_true, y_pred)), NMSE


def getBioImageEdgeM(intersected, parcel_dim):
    """
    Construct all plots and summary values for output from the executed CPM
    :return:
    """
    # For best performing bin (can be r or p):
    # BioImage.csv binary matrix
    with open('/1d-2d-map.npy', 'rb') as f:
        mapping = np.load(f, allow_pickle=True)
    # Binary matrix B
    B = np.zeros(shape=(parcel_dim,parcel_dim), dtype=int)
    for i in intersected:
            x,y = mapping[i]
            B[x,y] = 1
            B[y,x] = 1
    return B

