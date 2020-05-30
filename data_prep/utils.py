import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import scipy.stats as ss 
from scipy.stats import norm, skew

def scatter_plot(df, column1, column2):
    fig, ax = plt.subplots()
    ax.scatter(x = df[column1], y = df[column2])
    plt.ylabel(column2, fontsize=13)
    plt.xlabel(column1, fontsize=13)
    plt.show()

def norm_target(df, target):
    fig, ax = plt.subplots()
    sns.set(); np.random.seed(0)
    ax = sns.distplot(df[target] , fit=norm)
    (mu, sigma) = norm.fit(df[target])

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
    plt.ylabel('Frequency')
    plt.title(target + ' distribution')
    plt.show()

def qq_plot(df, target):
    fig = plt.figure()
    res = ss.probplot(df[target], plot=plt)
    plt.show()

def data_corr(df):
    corrmat = df.corr()
    sns.set(); np.random.seed(0)
    ax = sns.heatmap(corrmat, vmax=0.9, square=True)
    plt.show()
    