"""

@author Rylan Marianchuk
May 6, 2021
"""
import sys
from CPM_Help import *
import CPM_Cythoned
from glmnet_python.glmnet_python import cvglmnetCoef, cvglmnet
from sklearn.linear_model import LinearRegression
import time
import pandas as pd

class CPM:

    def __init__(self, FC, Y, Z, iterations, parcel, wdir):
        """
        :param FC: n by p Functional Connectivity Matrix, flattened. Each row is a subjects FC
        :param Y: n by 1 target phenotype vector
        :param Z: matrix of controls to regress out on model evaluation, can be None type
        :param iterations: amount of times to run 10-fold Cross-Validation
        :param parcel: the dimensionality of the parcelation p = (parcel*parcel - parcel) / 2 (entries in lower triangle)
        :param wdir: the directory where results should be stored
        """
        # dimension, side length of square connectome matrix, parcel granularity
        self.parcel = parcel
        # Number of times to run cross validation
        self.iterations = iterations
        self.wdir = wdir

        # Matrix of columns vectors holding variables to regress out in partial corr during model evaluation
        self.Z = Z

        self.FC = FC
        self.target = Y

        # n subjects
        self.n = len(FC)
        # p predictors
        self.p = len(FC[0])


    def apply10fold(self, interval, pselect):
        """
        :param interval = (lower_thresh, upper_thresh)
        :param pselect bool whether select edges by pearson significance values (p) or r values
        :return: iteratable of lists holding the errors for each model:
                 e_ridge, e_EN, e_lasso, e_lsq,
                 then the binary edge selection matrix
                 B,
                 then the dimensionality by bin data
                 dims
        """
        if self.n < 90: folds = 5
        else: folds = 10
        cv_size = self.n // folds + 1

        # Error containers
        e_ridge = []
        e_EN = []
        e_lasso = []
        e_lsq = []

        # To intersect all edges chosen
        #all_edges = []
        # Count all dimensionalities
        dims = []
        for _ in range(self.iterations):
            # Use index masking for cross-validation
            # Get random list of 100 integers
            ind = np.arange(0, self.n)
            np.random.shuffle(ind)
            y_true = np.array([])
            y_pred_ridge = np.array([])
            y_pred_EN = np.array([])
            y_pred_lasso = np.array([])
            y_pred_lsq = np.array([])
            for i in range(folds):
                # Cut the indexes by 10 each time
                test_ind = ind[cv_size * i:cv_size * (i + 1)]
                #print("Test indexes: from index " + str(cv_size * i) + " to " + str(cv_size * (i + 1)))
                #print("Length of test_ind:" + str(len(test_ind)))
                #print("Length of y_true:" + str(len(y_true)))
                train_ind = np.array(list(set(ind) - set(test_ind)))
                np.random.shuffle(train_ind)

                # Populate training data from edges chosen, call Cython here
                FC_flat = self.FC[train_ind]
                # Select by r or p
                inds, dim = CPM_Cythoned.get_Xtr(FC_flat, self.target[train_ind],
                                                 interval[0], interval[1], pselect)
                X_tr = FC_flat[:,inds]
                # Ensure we have enough edges to continue after masking
                if dim < 1:
                    print("Zero edges masked on interval " + str(interval) + " of fold " + str(i) + " of iteration " + str(_) +
                          ", consider the intervals chosen. Your interval doesn't capture enough edges")
                    return

                #all_edges.append(inds)
                dims.append(dim)

                # FIT MODELS HERE USING GLMNET---------------------
                ridge = cvglmnet(x=X_tr, y=self.target[train_ind], alpha=0)
                R_coefs = cvglmnetCoef(ridge, s='lambda_min')
                betas_ridge, intercept_ridge = R_coefs[1:].flatten(), float(R_coefs[0])

                lasso = cvglmnet(x=X_tr, y=self.target[train_ind], alpha=1)
                L_coefs = cvglmnetCoef(lasso, s='lambda_min')
                betas_lasso, intercept_lasso = L_coefs[1:].flatten(), float(L_coefs[0])

                en = cvglmnet(x=X_tr, y=self.target[train_ind], alpha=0.5)
                EN_coefs = cvglmnetCoef(en, s='lambda_min')
                betas_EN, intercept_EN = EN_coefs[1:].flatten(), float(EN_coefs[0])

                lsq = LinearRegression().fit(X_tr, self.target[train_ind])
                betas_lsq, intercept_lsq = lsq.coef_, lsq.intercept_

                # Add y predictions
                X_te = self.FC[test_ind]
                X_te = X_te[:,inds]

                y_true = np.concatenate((y_true, self.target[test_ind]))

                y_pred_ridge = np.concatenate((y_pred_ridge, X_te.dot(betas_ridge) + intercept_ridge))
                y_pred_EN = np.concatenate((y_pred_EN, X_te.dot(betas_EN) + intercept_EN))
                y_pred_lasso = np.concatenate((y_pred_lasso, X_te.dot(betas_lasso) + intercept_lasso))
                y_pred_lsq = np.concatenate((y_pred_lsq, X_te.dot(betas_lsq) + intercept_lsq))

            e_ridge.append(partial_corr(y_pred_ridge, y_true, Z=self.Z))
            e_EN.append(partial_corr(y_pred_EN, y_true, Z=self.Z))
            e_lasso.append(partial_corr(y_pred_lasso, y_true, Z=self.Z))
            e_lsq.append(partial_corr(y_pred_lsq, y_true, Z=self.Z))

        # Save the bioimage .csv of edges common across iterations and folds

        #intersected = reduce(np.intersect1d, all_edges)
        #print("Intersected: " + str(intersected))
        # Take the top 50% most frequently masked edges
        #u, count = np.unique(np.array(all_edges).flatten(), return_counts=True)
        #B = getBioImageEdgeM(u[np.argsort(-count)][:len(u)//2], self.parcel)
        B = []
        return e_ridge, e_EN, e_lasso, e_lsq, B, dims


    def iterate_r_bins(self, neg_edge, cap=0.6):
        """
        Call apply10fold() over r selections intervals
        :param neg_edge: bool, whether to take negative edge or positive
        :param cap: the max bin value

        Save results into specifically named files, where summary.py uses the same names
        DO NOT modify the file names! summary.py uses them!
        """
        bins = []
        for upper in np.linspace(0.025, 1, 40):
            upper = round(upper, ndigits=3)
            lower = round(upper - 0.025, ndigits=3)
            if upper > cap: break
            if neg_edge:
                bins.append((-upper, -lower))
            else:
                bins.append((lower, upper))

        results_ridge = []
        results_EN = []
        results_lasso = []
        results_lsq = []
        results_dims = []
        for bin in bins:
            V = self.apply10fold(bin, False)
            if V is None: break
            e_ridge, e_EN, e_lasso, e_lsq, B, dims = V
            results_ridge.append([bin, e_ridge])
            results_EN.append([bin, e_EN])
            results_lasso.append([bin, e_lasso])
            results_lsq.append([bin, e_lsq])
            results_dims.append([bin, dims])
            #np.savetxt("../results/bioimage-edges-" + str(bin) + "-rpos=" + str(not neg_edge) + ".csv", B, delimiter=",")

        # Save these as .npy files
        with open(wdir + "/results/npy/ridge-errs-rpos=" + str(not neg_edge) + ".npy", 'wb') as f:
            np.save(f, np.array(results_ridge))
        with open(wdir + "/results/npy/elasticnet-errs-rpos=" + str(not neg_edge) + ".npy", 'wb') as f:
            np.save(f, np.array(results_EN))
        with open(wdir + "/results/npy/lasso-errs-rpos=" + str(not neg_edge) + ".npy", 'wb') as f:
            np.save(f, np.array(results_lasso))
        with open(wdir + "/results/npy/ols-errs-rpos=" + str(not neg_edge) + ".npy", 'wb') as f:
            np.save(f, np.array(results_lsq))
        with open(wdir + "/results/npy/dimensionality-rpos=" + str(not neg_edge) + ".npy", 'wb') as f:
            np.save(f, np.array(results_dims))

    def iterate_p_bins(self):
        """
        Call apply10fold() over p selections intervals of 0.01 and 0.05

        Save results into specifically named files, where summary.py uses the same names
        DO NOT modify the file names! summary.py uses them!
        """
        bins = [(0, 0.01), (0, 0.05)]

        results_ridge = []
        results_EN = []
        results_lasso = []
        results_lsq = []
        results_dims = []
        for bin in bins:
            V = self.apply10fold(bin, True)
            if V is None: break
            e_ridge, e_EN, e_lasso, e_lsq, B, dims = V
            results_ridge.append([bin, e_ridge])
            results_EN.append([bin, e_EN])
            results_lasso.append([bin, e_lasso])
            results_lsq.append([bin, e_lsq])
            results_dims.append([bin, dims])
            #np.savetxt("../results/bioimage-edges-" + str(bin) + "-pvalue.csv", B, delimiter=",")

        # Save these as .npy files
        with open(wdir + "/results/npy/ridge-errs-pvalue.npy", 'wb') as f:
            np.save(f, np.array(results_ridge))
        with open(wdir + "/results/npy/elasticnet-errs-pvalue.npy", 'wb') as f:
            np.save(f, np.array(results_EN))
        with open(wdir + "/results/npy/lasso-errs-pvalue.npy", 'wb') as f:
            np.save(f, np.array(results_lasso))
        with open(wdir + "/results/npy/ols-errs-pvalue.npy", 'wb') as f:
            np.save(f, np.array(results_lsq))
        with open(wdir + "/results/npy/dimensionality-pvalue.npy", 'wb') as f:
            np.save(f, np.array(results_dims))


start = time.time()

print("Multivariate Iterations requested: " + sys.argv[1])
print("Multivariate Parcellation dimension of the FC matrix: " + sys.argv[2])
wdir = sys.argv[3]
print("wdir:  " + wdir)

try:
    Z = np.array(pd.read_csv(wdir + "/control.csv"))
except Exception:
    Z = None

cpm = CPM(FC=np.load(wdir + "/FC_flat.npy"), Y=np.load(wdir + "/target.npy"),
          Z=Z, iterations=int(sys.argv[1]), parcel=int(sys.argv[2]), wdir=wdir)
# Run positive r edge bins
cpm.iterate_r_bins(neg_edge=False)
print("Multivariate finished positive r bins in " + str(time.time() - start) + " seconds.")
# Run negative r edge bins
cpm.iterate_r_bins(neg_edge=True)
print("Multivariate finished negative r bins after " + str(time.time() - start) + " seconds.")
# Run p bins
cpm.iterate_p_bins()
print("Multivariate done all in " + str(time.time() - start) + " seconds.")
