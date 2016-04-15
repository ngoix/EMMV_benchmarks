import numpy as np

# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import sys
sys.path.insert(1, '/home/nicolas/Bureau/OCRF')

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc
from sklearn.utils import shuffle as sh
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.datasets import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from sklearn.datasets import fetch_pendigits, fetch_pima, fetch_wilt
from sklearn.datasets import fetch_internet_ads, fetch_adult
from em import EM, MV  # , EM_approx, MV_approx, MV_approx_over
from sklearn.preprocessing import LabelBinarizer

np.random.seed(1)

# TODO: find good default parameters for every datasets
# TODO: make an average of ROC curves over 10 experiments
# TODO: idem in bench_lof, bench_isolation_forest (to be launch from master)
#       bench_ocsvm (to be created), bench_ocrf (to be created)

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt',  # 'internet_ads',
#             'adult']

# # continuous datasets:
# datasets = ['http', 'smtp', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']
# new: ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#       'pima', 'wilt', 'adult']

datasets = ['http', 'smtp', 'shuttle', 'spambase',
            'pendigits', 'pima', 'wilt', 'adult']


for dat in datasets:
    print 'dataset:', dat
    # loading and vectorization
    print('loading data')

    if dat == 'adult':
        dataset = fetch_adult(shuffle=True)
        X = dataset.data
        y = dataset.target
        # anormal data are those with label >50K:
        y = np.all((y != ' <=50K', y != ' <=50K.'), axis=0).astype(int)

    if dat == 'internet_ads':  # not adapted to oneclassrf
        dataset = fetch_internet_ads(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'ad.').astype(int)

    if dat == 'wilt':
        dataset = fetch_wilt(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 'w').astype(int)

    if dat == 'pima':
        dataset = fetch_pima(shuffle=True)
        X = dataset.data
        y = dataset.target

    if dat == 'pendigits':
        dataset = fetch_pendigits(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y == 4).astype(int)
        # anomalies = class 4

    if dat == 'arrhythmia':
        dataset = fetch_arrhythmia(shuffle=True)
        X = dataset.data
        y = dataset.target
        # rm 5 features containing some '?' (XXX to be mentionned in paper)
        X = np.delete(X, [10, 11, 12, 13, 14], axis=1)
        y = (y != 1).astype(int)
        # normal data are then those of class 1

    if dat == 'annthyroid':
        dataset = fetch_annthyroid(shuffle=True)
        X = dataset.data
        y = dataset.target
        y = (y != 3).astype(int)
        # normal data are then those of class 3

    if dat == 'spambase':
        dataset = fetch_spambase(shuffle=True)
        X = dataset.data
        y = dataset.target

    if dat == 'ionosphere':
        dataset = fetch_mldata('ionosphere')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y)
        y = (y != 1).astype(int)

    if dat in ['http', 'smtp', 'SA', 'SF']:
        dataset = fetch_kddcup99(subset=dat, shuffle=True, percent10=False)
        X = dataset.data
        y = dataset.target

    if dat == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        X, y = sh(X, y)
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    if dat == 'forestcover':
        dataset = fetch_covtype(shuffle=True)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        y = (y != 2).astype(int)

    print('vectorizing data')

    if dat == 'SF':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != 'normal.').astype(int)

    if dat == 'SA':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        lb.fit(X[:, 2])
        x2 = lb.transform(X[:, 2])
        lb.fit(X[:, 3])
        x3 = lb.transform(X[:, 3])
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != 'normal.').astype(int)

    if dat == 'http' or dat == 'smtp':
        y = (y != 'normal.').astype(int)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X = X.astype(float)
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    # fit:
    print('model processing...')
    # model = LocalOutlierFactor(n_neighbors=20)
    model = IsolationForest()
    # model = OneClassSVM()
    model.fit(X_train)
    max_axis = model.predict(X_test).max()
    axis_t = np.arange(0, max_axis, max_axis/1000.)
    axis_alpha = np.arange(0, 1, 0.001)

    plt.figure(figsize=(25, 13))
    # EM:
    plt.subplot(121)
    print 'em_'
    em_ = EM(model, X_test, axis_t)
    AUC = auc(axis_t, em_)
    plt.plot(axis_t, em_, lw=1,
             label='%s (area = %0.3f)'
             % (dat, AUC))

    # plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('t')
    plt.ylabel('EM(t)')
    plt.title('Excess-Mass curve')
    plt.legend(loc="lower left")

    # MV:
    plt.subplot(122)
    print 'mv_'
    mv_ = MV(model, X_test, axis_alpha)
    AUC = auc(axis_alpha, mv_)
    plt.plot(axis_alpha, mv_, lw=1,
             label='%s (area = %0.3f)'
             % (dat, AUC))

    plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 100])
    plt.xlabel('alpha')
    plt.ylabel('MV(alpha)')
    plt.title('Mass-Volume Curve')
    plt.legend(loc="upper left")

    plt.savefig('iforest_em_mv_' + dat)
