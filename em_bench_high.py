import numpy as np
import pdb
# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import sys
sys.path.insert(1, '/home/nicolas/Bureau/OCRF')


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle as sh
from sklearn.datasets import fetch_kddcup99, fetch_covtype, fetch_mldata
from sklearn.datasets import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from sklearn.datasets import fetch_pendigits, fetch_pima, fetch_wilt
from sklearn.datasets import fetch_internet_ads, fetch_adult
from em import em, mv  # , EM_approx, MV_approx, MV_approx_over
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

averaging = 20
max_features = 5
n_generated = 500000

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

# continuous datasets:
# datasets = ['http', 'smtp', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']
# new: ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
#       'pima', 'wilt', 'adult']

# datasets = [# 'http',
#             'smtp', 'shuttle', # 'spambase',
#             'pendigits', 'pima', 'wilt', 'adult']
datasets = ['forestcover', 'ionosphere', 'spambase', 'annthyroid', 'arrhythmia']

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

    # ### 10 % of abnormal max: ###
    index_normal = (y == 0)
    index_abnormal = (y == 1)
    if index_abnormal.sum() > 0.1 * index_normal.sum():
        X_normal = X[index_normal]
        X_abnormal = X[index_abnormal]
        n_anomalies = X_abnormal.shape[0]
        n_anomalies_max = int(0.1 * index_normal.sum())
        r = sh(np.arange(n_anomalies))[:n_anomalies_max]
        X = np.r_[X_normal, X_abnormal[r]]
        y = np.array([0] * X_normal.shape[0] + [1] * n_anomalies_max)
    # ######

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X = X.astype(float)
    X = scale(X)
    
    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    # # training only on normal data:
    # X_train = X_train[y_train == 0]
    # y_train = y_train[y_train == 0]

    # define models:
    iforest = IsolationForest()
    lof = LocalOutlierFactor(n_neighbors=20)
    ocsvm = OneClassSVM()

    n_samples, n_features = X_test.shape
    em_iforest, mv_iforest = 0, 0
    em_lof, mv_lof = 0, 0
    em_ocsvm, mv_ocsvm = 0, 0

    for nb_exp in range(averaging):
        features = shuffle(np.arange(n_features))[:max_features]
        X_train_ = X_train[:, features]
        X_ = X_test[:, features]

        lim_inf = X_.min(axis=0)
        lim_sup = X_.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()
        t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
        axis_alpha = np.arange(0.9, 0.99, 0.001)
        unif = np.random.uniform(lim_inf, lim_sup,
                                 size=(n_generated, max_features))

        iforest.fit(X_train_)
        lof.fit(X_train_)
        ocsvm.fit(X_train_[:min(100000, n_samples_train - 1)])
        print 'end of ocsvm training!'
        s_X_iforest = iforest.decision_function(X_)
        s_X_lof = lof.decision_function(X_)
        s_X_ocsvm = ocsvm.decision_function(X_).reshape(1, -1)[0]

        s_unif_iforest = iforest.decision_function(unif)
        s_unif_lof = lof.decision_function(unif)
        s_unif_ocsvm = ocsvm.decision_function(unif).reshape(1, -1)[0]

        em_iforest += em(t, n_samples_test, volume_support, s_unif_iforest,
                         s_X_iforest, n_generated)[0]
        mv_iforest += mv(axis_alpha, n_samples_test, volume_support, s_unif_iforest,
                         s_X_iforest, n_generated)[0]
        em_lof += em(t, n_samples_test, volume_support, s_unif_lof, s_X_lof,
                     n_generated)[0]
        mv_lof += mv(axis_alpha, n_samples_test, volume_support, s_unif_lof,
                     s_X_lof, n_generated)[0]
        em_ocsvm += em(t, n_samples_test, volume_support, s_unif_ocsvm,
                       s_X_ocsvm, n_generated)[0]
        mv_ocsvm += mv(axis_alpha, n_samples_test, volume_support, s_unif_ocsvm,
                       s_X_ocsvm, n_generated)[0]

    em_iforest /= averaging
    mv_iforest /= averaging
    em_lof /= averaging
    mv_lof /= averaging
    em_ocsvm /= averaging
    mv_ocsvm /= averaging

    with open('results_workshop/result_em_bench_high_unsupervised_with095_withano10percent_with_scale' + dat + '_'
              + str(max_features) + '_' +
              str(averaging) + '_' + '.txt', 'a') as result:
        result.write('em_iforest = ' + str(em_iforest) + '\n')
        result.write('em_lof = ' + str(em_lof) + '\n')
        result.write('em_ocsvm = ' + str(em_ocsvm) + '\n \n')

        result.write('mv_iforest = ' + str(mv_iforest) + '\n')
        result.write('mv_lof = ' + str(mv_lof) + '\n')
        result.write('mv_ocsvm = ' + str(mv_ocsvm) + '\n')
