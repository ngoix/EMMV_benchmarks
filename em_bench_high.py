import numpy as np

# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(1, '/home/nicolas/Bureau/OCRF')

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.utils import shuffle as sh
from sklearn.datasets import one_class_data

from em import em, mv

# framework: outlier or novelty detection
novelty_detection = True

# parameters of the algorithm:
averaging = 50
max_features = 5
n_generated = 100000
alpha_min = 0.9
alpha_max = 0.999
t_max = 0.9
ocsvm_max_train = 10000

np.random.seed(1)

# # datasets available:
# datasets = ['http', 'smtp', 'SA', 'SF', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt',  # 'internet_ads',
#             'adult']

# continuous datasets:
# datasets = ['http', 'smtp', 'shuttle', 'forestcover',
#             'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'pendigits', 'pima', 'wilt', 'adult']

# # high-dim continuous datasets:
# datasets = ['ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
#             'forestcover', 'shuttle', 'pendigits']

datasets = ['ionosphere']

for dat in datasets:
    # loading and vectorization
    X, y = one_class_data(dat)
    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    if novelty_detection:
        # training and testing only on normal data:
        X_train = X_train[y_train == 0]
        y_train = y_train[y_train == 0]
        X_test = X_test[y_test == 0]
        y_test = y_test[y_test == 0]

    # define models:
    iforest = IsolationForest()
    lof = LocalOutlierFactor(n_neighbors=20)
    ocsvm = OneClassSVM()

    n_samples, n_features = X_test.shape
    em_iforest, mv_iforest = 0, 0
    em_lof, mv_lof = 0, 0
    em_ocsvm, mv_ocsvm = 0, 0
    nb_exp = 0
    while nb_exp < averaging:
        features = sh(np.arange(n_features))[:max_features]
        X_train_ = X_train[:, features]
        X_ = X_test[:, features]

        lim_inf = X_.min(axis=0)
        lim_sup = X_.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()
        if volume_support > 0:
            nb_exp += 1
            t = np.arange(0, 100 / volume_support, 0.001 / volume_support)
            axis_alpha = np.arange(alpha_min, alpha_max, 0.001)
            unif = np.random.uniform(lim_inf, lim_sup,
                                     size=(n_generated, max_features))

            iforest.fit(X_train_)
            lof.fit(X_train_)
            ocsvm.fit(X_train_[:min(ocsvm_max_train, n_samples_train - 1)])
            print 'end of ocsvm training!'
            s_X_iforest = iforest.decision_function(X_)
            s_X_lof = lof.decision_function(X_)
            s_X_ocsvm = ocsvm.decision_function(X_).reshape(1, -1)[0]

            s_unif_iforest = iforest.decision_function(unif)
            s_unif_lof = lof.decision_function(unif)
            s_unif_ocsvm = ocsvm.decision_function(unif).reshape(1, -1)[0]

            em_iforest += em(t, t_max, volume_support, s_unif_iforest,
                             s_X_iforest, n_generated)[0]
            mv_iforest += mv(axis_alpha, volume_support, s_unif_iforest,
                             s_X_iforest, n_generated)[0]
            em_lof += em(t, t_max, volume_support, s_unif_lof, s_X_lof,
                         n_generated)[0]
            mv_lof += mv(axis_alpha, volume_support, s_unif_lof,
                         s_X_lof, n_generated)[0]
            em_ocsvm += em(t, t_max, volume_support, s_unif_ocsvm,
                           s_X_ocsvm, n_generated)[0]
            mv_ocsvm += mv(axis_alpha, volume_support, s_unif_ocsvm,
                           s_X_ocsvm, n_generated)[0]

    em_iforest /= averaging
    mv_iforest /= averaging
    em_lof /= averaging
    mv_lof /= averaging
    em_ocsvm /= averaging
    mv_ocsvm /= averaging

    with open('result_em_bench_high_unsupervised_with' + str(alpha_min) + '_factorized_with' + str(ocsvm_max_train) + 'ocsvm_' + dat + '_'
              + str(max_features) + '_' +
              str(averaging) + '_' + '.txt', 'a') as result:
        result.write('em_iforest = ' + str(em_iforest) + '\n')
        result.write('em_lof = ' + str(em_lof) + '\n')
        result.write('em_ocsvm = ' + str(em_ocsvm) + '\n \n')

        result.write('mv_iforest = ' + str(mv_iforest) + '\n')
        result.write('mv_lof = ' + str(mv_lof) + '\n')
        result.write('mv_ocsvm = ' + str(mv_ocsvm) + '\n')
