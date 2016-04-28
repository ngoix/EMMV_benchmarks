import numpy as np
import pdb
# import matplotlib.pyplot as plt
# for the cluster to save the fig:
import sys
sys.path.insert(1, '/home/nicolas/Bureau/OCRF')


import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from sklearn.datasets import one_class_data

from em import em, mv  # , EM_approx, MV_approx, MV_approx_over

n_generated = 100000
alpha_min = 0.9
alpha_max = 0.999
t_max = 0.9
ocsvm_max_train = 10000
# wilt: prendre t_max = 0.995, alpha_min = 0.995, alpha_max=0.999 (sans scale)
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

datasets = ['http', 'smtp',  # 'shuttle', 'pendigits',
            'pima', 'wilt', 'adult']

for dat in datasets:
    plt.clf()
    plt.figure(figsize=(25, 13))

    # loading and vectorization
    X, y = one_class_data(dat)

    n_samples, n_features = np.shape(X)
    n_samples_train = n_samples // 2
    n_samples_test = n_samples - n_samples_train

    X_train = X[:n_samples_train, :]
    X_test = X[n_samples_train:, :]
    y_train = y[:n_samples_train]
    y_test = y[n_samples_train:]

    # training and testing only on normal data:
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]
    X_test = X_test[y_test == 0]
    y_test = y_test[y_test == 0]

    # define models:
    iforest = IsolationForest()
    lof = LocalOutlierFactor(n_neighbors=20)
    ocsvm = OneClassSVM()

    lim_inf = X.min(axis=0)
    lim_sup = X.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup,
                             size=(n_generated, n_features))

    # fit:
    print('IsolationForest processing...')
    iforest = IsolationForest()
    iforest.fit(X_train)
    s_X_iforest = iforest.decision_function(X_test)
    print('LocalOutlierFactor processing...')
    lof = LocalOutlierFactor(n_neighbors=20)
    lof.fit(X_train)
    s_X_lof = lof.decision_function(X_test)
    print('OneClassSVM processing...')
    ocsvm = OneClassSVM()
    ocsvm.fit(X_train[:min(ocsvm_max_train, n_samples_train - 1)])
    s_X_ocsvm = ocsvm.decision_function(X_test).reshape(1, -1)[0]
    s_unif_iforest = iforest.decision_function(unif)
    s_unif_lof = lof.decision_function(unif)
    s_unif_ocsvm = ocsvm.decision_function(unif).reshape(1, -1)[0]
    plt.subplot(121)
    auc_iforest, em_iforest, amax_iforest = em(t, t_max,
                                               volume_support,
                                               s_unif_iforest,
                                               s_X_iforest, n_generated)

    auc_lof, em_lof, amax_lof = em(t, t_max, volume_support,
                                   s_unif_lof, s_X_lof, n_generated)

    auc_ocsvm, em_ocsvm, amax_ocsvm = em(t, t_max, volume_support,
                                         s_unif_ocsvm, s_X_ocsvm,
                                         n_generated)
    if amax_iforest == -1 or amax_lof == -1 or amax_ocsvm == -1:
        amax = -1
    else:
        amax = max(amax_iforest, amax_lof, amax_ocsvm)
    plt.subplot(121)
    plt.plot(t[:amax], em_iforest[:amax], lw=1,
             label='%s (em_score = %0.3e)'
             % ('iforest', auc_iforest))
    plt.plot(t[:amax], em_lof[:amax], lw=1,
             label='%s (em-score = %0.3e)'
             % ('lof', auc_lof))
    plt.plot(t[:amax], em_ocsvm[:amax], lw=1,
             label='%s (em-score = %0.3e)'
             % ('ocsvm', auc_ocsvm))

    plt.ylim([-0.05, 1.05])
    plt.xlabel('t', fontsize=20)
    plt.ylabel('EM(t)', fontsize=20)
    plt.title('Excess-Mass curve for ' + dat + ' dataset', fontsize=20)
    plt.legend(loc="lower right")

    plt.subplot(122)
    print 'mv_iforest'
    auc_iforest, mv_iforest = mv(axis_alpha, volume_support,
                                 s_unif_iforest, s_X_iforest, n_generated)
    auc_lof, mv_lof = mv(axis_alpha, volume_support,
                         s_unif_lof, s_X_lof, n_generated)
    auc_ocsvm, mv_ocsvm = mv(axis_alpha, volume_support,
                             s_unif_ocsvm, s_X_ocsvm, n_generated)
    plt.plot(axis_alpha, mv_iforest, lw=1,
             label='%s (mv-score = %0.3e)'
             % ('iforest', auc_iforest))
    plt.plot(axis_alpha, mv_lof, lw=1,
             label='%s (mv-score = %0.3e)'
             % ('lof', auc_lof))
    plt.plot(axis_alpha, mv_ocsvm, lw=1,
             label='%s (mv-score = %0.3e)'
             % ('ocsvm', auc_ocsvm))

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 100])
    plt.xlabel('alpha', fontsize=20)
    plt.ylabel('MV(alpha)', fontsize=20)
    plt.title('Mass-Volume Curve for ' + dat + ' dataset', fontsize=20)
    plt.legend(loc="upper left")

    # plt.savefig('unsup_mv_em_' + dat + '_unsupervised_09_factorized')
    # plt.savefig('mv_em_' + dat + '_unsupervised_09_factorized')

    # plt.savefig('mv_em_' + dat + '_supervised'
    #             + '_alphamin' + str(int(100 * alpha_min)) + '_'
    #             + '_n_generated' + str(n_generated) + '_'
    #             + '_ocsvm' + str(ocsvm_max_train) + '_'
    #             + '_factorized_pruning')

    plt.show()
