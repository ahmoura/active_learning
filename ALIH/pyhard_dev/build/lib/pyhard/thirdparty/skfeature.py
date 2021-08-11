"""
Code from scikit-feature https://github.com/jundongl/scikit-feature/tree/48cffad4e88ff4b9d2f1c7baffb314d1b3303792

Feature selection repository scikit-feature in Python.
scikit-feature is an open-source feature selection repository in Python
developed by Data Mining and Machine Learning Lab at Arizona State University

http://featureselection.asu.edu/

Copyright (c) 2020 Jundong Li
"""

import numpy as np
from . import entropy_estimators as ee


def lcsi(X, y, **kwargs):
    """
    This function implements the basic scoring criteria for linear combination of shannon information term.
    The scoring criteria is calculated based on the formula j_cmi=I(f;y)-beta*sum_j(I(fj;f))+gamma*sum(I(fj;f|y))

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        Parameters for different feature selection algorithms.
        beta: {float}
            beta is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
        gamma: {float}
            gamma is the parameter in j_cmi=I(f;y)-beta*sum(I(fj;f))+gamma*sum(I(fj;f|y))
        function_name: {string}
            name of the feature selection function
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape: (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features

    f_select = beta = gamma = idx = None

    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
    else:
        n_selected_features = None

    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = ee.mi(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if n_selected_features is not None:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break

        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        if 'function_name' in kwargs.keys():
            if kwargs['function_name'] == 'MRMR':
                beta = 1.0 / len(F)
            elif kwargs['function_name'] == 'JMI':
                beta = 1.0 / len(F)
                gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += ee.mi(f_select, f)
                t3[i] += ee.cmi(f_select, f, y)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta*t2[i] + gamma*t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F), np.array(J_CMI), np.array(MIfy)


def cmim(X, y, **kwargs):
    """
    This function implements the CMIM feature selection.
    The scoring criteria is calculated based on the formula j_cmim=I(f;y)-max_j(I(fj;f)-I(fj;f|y))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        Input data, guaranteed to be a discrete numpy array
    y: {numpy array}, shape (n_samples,)
        guaranteed to be a numpy array
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMIM: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape

    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMIM = []
    # Mutual information between feature and response
    MIfy = []
    f_select = None
    idx = None

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
    else:
        n_selected_features = None

    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)

    # max stores max(I(fj;f)-I(fj;f|y)) for each feature f
    # we assign an extreme small value to max[i] ito make it is smaller than possible value of max(I(fj;f)-I(fj;f|y))
    vmax = -1e7 * np.ones(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = ee.mi(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_cmim = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMIM.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if n_selected_features is not None:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmim <= 0:
                break

        # we assign an extreme small value to j_cmim to ensure it is smaller than all possible values of j_cmim
        j_cmim = -1e12
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = ee.mi(f_select, f)
                t3 = ee.cmi(f_select, f, y)
                if t2 - t3 > vmax[i]:
                    vmax[i] = t2 - t3
                # calculate j_cmim for feature i (not in F)
                t = t1[i] - vmax[i]
                # record the largest j_cmim and the corresponding feature index
                if t > j_cmim:
                    j_cmim = t
                    idx = i
        F.append(idx)
        J_CMIM.append(j_cmim)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F), np.array(J_CMIM), np.array(MIfy)


def mim(X, y, **kwargs):
    """
    This function implements the MIM feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, beta=0, gamma=0, n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, beta=0, gamma=0)
    return F, J_CMI, MIfy


def mifs(X, y, **kwargs):
    """
    This function implements the MIFS feature selection

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response

    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    if 'beta' not in kwargs.keys():
        beta = 0.5
    else:
        beta = kwargs['beta']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, beta=beta, gamma=0, n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, beta=beta, gamma=0)
    return F, J_CMI, MIfy


def mrmr(X, y, **kwargs):
    """
    This function implements the MRMR feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, gamma=0, function_name='MRMR')
    return F, J_CMI, MIfy


def cife(X, y, **kwargs):
    """
    This function implements the CIFE feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, beta=1, gamma=1, n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, beta=1, gamma=1)
    return F, J_CMI, MIfy


def jmi(X, y, **kwargs):
    """
    This function implements the JMI feature selection
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_CMI: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI')
    return F, J_CMI, MIfy


def disr(X, y, **kwargs):
    """
    This function implement the DISR feature selection.
    The scoring criteria is calculated based on the formula j_disr=sum_j(I(f,fj;y)/H(f,fj,y))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features, )
        index of selected features, F[0] is the most important feature
    J_DISR: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    Reference
    ---------
    Brown, Gavin et al. "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature
    Selection." JMLR 2012.
    """

    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_DISR = []
    # Mutual information between feature and response
    MIfy = []

    f_select = idx = t1 = None

    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
    else:
        n_selected_features = None

    # vsum stores sum_j(I(f,fj;y)/H(f,fj,y)) for each feature f
    vsum = np.zeros(n_features)

    # make sure that j_cmi is positive at the very beginning
    j_disr = 1

    while True:
        if len(F) == 0:
            # t1 stores I(f;y) for each feature f
            t1 = np.zeros(n_features)
            for i in range(n_features):
                f = X[:, i]
                t1[i] = ee.mi(f, y)
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_DISR.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if n_selected_features is not None:
            if len(F) == n_selected_features:
                break
        else:
            if j_disr <= 0:
                break

        # we assign an extreme small value to j_disr to ensure that it is smaller than all possible value of j_disr
        j_disr = -1E30
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = ee.mi(f_select, y) + ee.cmi(f, y, f_select)
                t3 = ee.entropy(f) + ee.centropy(f_select, f) + (ee.centropy(y, f_select) - ee.cmi(y, f, f_select))
                vsum[i] += np.true_divide(t2, t3)
                # record the largest j_disr and the corresponding feature index
                if vsum[i] > j_disr:
                    j_disr = vsum[i]
                    idx = i
        F.append(idx)
        J_DISR.append(j_disr)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F), np.array(J_DISR), np.array(MIfy)


def icap(X, y, **kwargs):
    """
    This function implements the ICAP feature selection.
    The scoring criteria is calculated based on the formula j_icap = I(f;y) - max_j(0,(I(fj;f)-I(fj;f|y)))
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be a discrete data matrix
    y: {numpy array}, shape (n_samples,)
        input class labels
    kwargs: {dictionary}
        n_selected_features: {int}
            number of features to select
    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    J_ICAP: {numpy array}, shape: (n_features,)
        corresponding objective function value of selected features
    MIfy: {numpy array}, shape: (n_features,)
        corresponding mutual information between selected features and response
    """
    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_ICAP = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
    else:
        n_selected_features = None

    f_select = idx = None

    # t1 contains I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # vmax contains max_j(0,(I(fj;f)-I(fj;f|y))) for each feature f
    vmax = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = ee.mi(f, y)

    # make sure that j_cmi is positive at the very beginning
    j_icap = 1

    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_ICAP.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]

        if n_selected_features is not None:
            if len(F) == n_selected_features:
                break
        else:
            if j_icap <= 0:
                break

        # we assign an extreme small value to j_icap to ensure it is smaller than all possible values of j_icap
        j_icap = -1e12
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2 = ee.mi(f_select, f)
                t3 = ee.cmi(f_select, f, y)
                if t2-t3 > vmax[i]:
                    vmax[i] = t2-t3
                # calculate j_icap for feature i (not in F)
                t = t1[i] - vmax[i]
                # record the largest j_icap and the corresponding feature index
                if t > j_icap:
                    j_icap = t
                    idx = i
        F.append(idx)
        J_ICAP.append(j_icap)
        MIfy.append(t1[idx])
        f_select = X[:, idx]

    return np.array(F), np.array(J_ICAP), np.array(MIfy)
