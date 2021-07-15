"""
Generate figures needed after all jobs complete,

Save figures to .html files
Output bioimage suite compatiable .csv to visualize edges

@author Rylan Marianchuk
rylan.marianchuk@ucalgary.ca
July 2021
"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import sys

def grid6_r(wdir, pos_edge, writehtml=False):
    """
    :param pos_edge:
    :return:
    """

    err_index2str = {
        0 : "Pearson",
        1 : "Spearman",
        2 : "Partial Corr",
        3 : "RMSE",
        4 : "NMSE",
    }

    if pos_edge:
        marker_col = "black"
        s = "r (+)"
    else:
        marker_col = "red"
        s = "r (-)"


    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Univariate " + s, "Ridge " + s, "Random Forest " + s,
                                        "LASSO " + s, "Elastic Net " + s, " Ordinary Least Squares " + s),
                        shared_yaxes=True)

    for x,file in enumerate((wdir + '/results/npy/univariate-errs-rpos=' + str(pos_edge) + ".npy", wdir + '/results/npy/ridge-errs-rpos=' + str(pos_edge) + ".npy",
                      wdir + '/results/npy/randomforest-errs-rpos=' + str(pos_edge) + ".npy")):
        with open(file, 'rb') as f:
            errs = np.load(f, allow_pickle=True)

            for i in range(len(errs)):
                fig.append_trace(go.Violin(y=errs[i, 1], name=str(errs[i, 0])[1:-1], marker_color=marker_col), 1, x+1)

    for x, file in enumerate((wdir + '/results/npy/lasso-errs-rpos=' + str(pos_edge) + ".npy", wdir + '/results/npy/elasticnet-errs-rpos=' + str(pos_edge) + ".npy",
                      wdir + '/results/npy/ols-errs-rpos=' + str(pos_edge) + ".npy")):
        with open(file, 'rb') as f:
            errs = np.load(f, allow_pickle=True)

            for i in range(len(errs)):
                fig.append_trace(go.Violin(y=errs[i, 1], name=str(errs[i, 0])[1:-1], marker_color=marker_col), 2, x+1)

    fig.update_traces(marker=dict(size=3.5), points='all', jitter=0.45)
    fig.update_layout(title="Performance Comparison of Prediction Methods<br>r Edge Masking", showlegend=False,
                      xaxis_title="r interval", yaxis_title="Correlation of Predicted vs True")
    if writehtml:
        fig.write_html(wdir + "/results/figures/" + s + "-performance-comparison.html")
    else:
        fig.show()
    return

def grid6_p(wdir, writehtml=False):
    """
    :return:
    """

    err_index2str = {
        0 : "Pearson r",
        1 : "Spearman r",
        2 : "Partial Corr",
        3 : "RMSE",
        4 : "NMSE",
    }


    s = "p-value"


    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=("Univariate " + s, "Ridge " + s, "Random Forest " + s,
                                        "LASSO " + s, "Elastic Net " + s, "Ordinary Least Squares " + s),
                        shared_yaxes=True)

    for x,file in enumerate((wdir + '/results/npy/univariate-errs-pvalue.npy', wdir + '/results/npy/ridge-errs-pvalue.npy',
                      wdir + '/results/npy/randomforest-errs-pvalue.npy')):
        with open(file, 'rb') as f:
            errs = np.load(f, allow_pickle=True)

            for i in range(len(errs)):
                fig.append_trace(go.Violin(y=errs[i, 1], name=str(errs[i, 0])[1:-1], marker_color='orange'), 1, x+1)

    for x, file in enumerate((wdir + '/results/npy/lasso-errs-pvalue.npy', wdir + '/results/npy/elasticnet-errs-pvalue.npy',
                      wdir + '/results/npy/ols-errs-pvalue.npy')):
        with open(file, 'rb') as f:
            errs = np.load(f, allow_pickle=True)

            for i in range(len(errs)):
                fig.append_trace(go.Violin(y=errs[i, 1], name=str(errs[i, 0])[1:-1], marker_color='orange'), 2, x+1)

    fig.update_traces(marker=dict(size=3.5), points='all', jitter=0.45)
    fig.update_layout(title="Performance Comparison of Prediction Methods<br>p-value Edge Masking", showlegend=False,
                      xaxis_title="p interval", yaxis_title="Correlation of Predicted vs True")

    if writehtml:
        fig.write_html(wdir + "/results/figures/" + s + "-performance-comparison.html")
    else:
        fig.show()
    return


def dimensionality(wdir, writehtml=False):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("r (+) bin dimensionality", "r (-) bin dimensionality", "p-value dimensionality"),
                        shared_yaxes=True)
    with open(wdir + "/results/npy/dimensionality-rpos=True.npy", 'rb') as f:
        dim = np.load(f, allow_pickle=True)

        for i in range(len(dim)):
            fig.append_trace(go.Violin(y=dim[i, 1], name=str(dim[i, 0])[1:-1], marker_color='black'), 1, 1)


    with open(wdir + "/results/npy/dimensionality-rpos=False.npy", 'rb') as f:
        dim = np.load(f, allow_pickle=True)

        for i in range(len(dim)):
            fig.append_trace(go.Violin(y=dim[i, 1], name=str(dim[i, 0])[1:-1], marker_color='red'), 1, 2)


    with open(wdir + "/results/npy/dimensionality-pvalue.npy", 'rb') as f:
        dim = np.load(f, allow_pickle=True)

        for i in range(len(dim)):
            fig.append_trace(go.Violin(y=dim[i, 1], name=str(dim[i, 0])[1:-1], marker_color='orange'), 1, 3)

    fig.update_traces(points='all', jitter=0.15, marker=dict(size=3.5))
    fig.update_layout(title="Edge counts<br>Dimensionality of fitted regression at each interval",
                      font=dict(size=18.5), showlegend=False)

    fig.update_yaxes(title_text="# edges masked", title_font={"size": 20})
    if writehtml:
        fig.write_html(wdir + "/results/figures/dimensionality-by-bin.html")
    else:
        fig.show()
    return

wdir = sys.argv[1]
grid6_r(wdir, pos_edge=True, writehtml=True)
grid6_r(wdir, pos_edge=False, writehtml=True)
grid6_p(wdir, writehtml=True)
dimensionality(wdir, writehtml=True)
