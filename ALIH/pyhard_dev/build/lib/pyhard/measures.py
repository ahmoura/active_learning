import itertools
import logging

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors, DistanceMetric

from . import get_seed
from .base import BaseMeasures


def minmax(f: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    assert len(classes) == 2
    c1 = classes[0]
    c2 = classes[1]
    return min(np.max(f[y == c1]), np.max(f[y == c2]))


def maxmin(f: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    assert len(classes) == 2
    c1 = classes[0]
    c2 = classes[1]
    return max(np.min(f[y == c1]), np.min(f[y == c2]))


def maxmax(f: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    assert len(classes) == 2
    c1 = classes[0]
    c2 = classes[1]
    return max(np.max(f[y == c1]), np.max(f[y == c2]))


def minmin(f: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    assert len(classes) == 2
    c1 = classes[0]
    c2 = classes[1]
    return min(np.min(f[y == c1]), np.min(f[y == c2]))


def gower_distance(X):
    """
    This function expects a pandas dataframe as input
    The data frame is to contain the features along the columns. Based on these features a
    distance matrix will be returned which will contain the pairwise gower distance between the rows
    All variables of object type will be treated as nominal variables and the others will be treated as
    numeric variables.
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable
    (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    N = len(X)
    n_feat = X.shape[1]
    cumsum_dist = np.zeros((N, N))

    for i in range(n_feat):
        feature = X.iloc[:, [i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature)
            feature_dist /= max(np.ptp(feature.values), 1e-8)

        cumsum_dist += feature_dist * 1 / n_feat

    return cumsum_dist


class ClassificationMeasures(BaseMeasures):
    """
    Hardness measures class. It provides separate methods to compute each measure

    :param data: a dataframe where each line is an instace and columns are features. One column should contain the
        labels. The name of the column with labels can be set with parameter `labels_col`
    :type: pandas DataFrame
    :param labels_col: name of the column that contains the labels of the instances (default None - uses the
        last column)
    :param ccp_alpha: pruning parameter for pruned tree measures. If none is passed, then it attempts to tune it
        automatically
    """

    _measures_dict = {
        'kDN': 'k_disagreeing_neighbors',
        'DS': 'disjunct_size',
        'DCP': 'disjunct_class_percentage',
        'TD_P': 'tree_depth_pruned',
        'TD_U': 'tree_depth_unpruned',
        'CL': 'class_likeliood',
        'CLD': 'class_likeliood_diff',
        'MV': 'minority_value',
        'CB': 'class_balance',
        'N1': 'borderline_points',
        'N2': 'intra_extra_ratio',
        'LSC': 'local_set_cardinality',
        'LSR': 'ls_radius',
        'Harmfulness': 'harmfulness',
        'Usefulness': 'usefulness',
        'F1': 'f1',
        'F2': 'f2',
        'F3': 'f3',
        'F4': 'f4'
    }

    logger = logging.getLogger(__name__)

    def __init__(self, data: pd.DataFrame, ccp_alpha=None, labels_col=None):
        if labels_col is None:
            self.labels_col = data.columns[-1]
            self.y = data.iloc[:, -1]
        else:
            self.labels_col = labels_col
            self.y = data[labels_col]
        self.data = data.reset_index(drop=True)
        self.X = data.drop(columns=self.labels_col)
        self.N = len(data)

        seed = get_seed()

        # Decision Tree Classifier
        dtc = tree.DecisionTreeClassifier(min_samples_split=2, criterion='gini', random_state=seed)
        self.dtc = dtc.fit(self.X, self.y)

        # Decision Tree Classifier Pruned
        if ccp_alpha is None:
            parameters = {'ccp_alpha': np.linspace(0.001, 0.1, num=100)}
            dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=seed)
            clf = GridSearchCV(dtc, parameters)
            clf = clf.fit(self.X, self.y)
            ccp_alpha = clf.best_params_['ccp_alpha']

        self.dtc_pruned = tree.DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alpha, random_state=seed)
        self.dtc_pruned = self.dtc_pruned.fit(self.X, self.y)

        # Gower distance matrix
        # self.dist_matrix_gower = gower.gower_matrix(self.X.values.copy())
        self.dist_matrix_gower = gower_distance(self.X)
        delta = np.diag(-np.ones(self.dist_matrix_gower.shape[0]))
        self.indices_gower = np.argsort(self.dist_matrix_gower + delta, axis=1)
        self.distances_gower = np.sort(self.dist_matrix_gower, axis=1)

        self.dot = None

    def k_disagreeing_neighbors(self, k=5, distance='gower'):
        data = self.data.copy()
        if distance == 'gower':
            indices = self.indices_gower[:, :k + 1]
        else:
            nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        kDN = []
        for i in range(0, len(data)):
            v = data.loc[indices[i]][self.labels_col].values
            kDN.append(np.sum(v[1:] != v[0]) / k)
        return np.array(kDN)

    def disjunct_size(self):
        data = self.data.copy()

        data['leaf_id'] = self.dtc.apply(self.X)
        df_count = data.groupby('leaf_id').count().iloc[:, 0].to_frame('count').subtract(1)
        data = data.join(df_count, on='leaf_id')
        DS = data['count'].divide(data['count'].max())

        return 1 - DS.values

    def disjunct_class_percentage(self):
        data = self.data.copy()

        data['leaf_id'] = self.dtc_pruned.apply(self.X)
        dcp = []
        for index, row in data.iterrows():
            df_leaf = data[data['leaf_id'] == row['leaf_id']]
            dcp.append(len(df_leaf[df_leaf[self.labels_col] == row[self.labels_col]]) / len(df_leaf))

        return 1 - np.array(dcp)

    def tree_depth_unpruned(self):
        return self.X.apply(lambda x: self.dtc.decision_path([x]).sum() - 1,
                            axis=1, raw=True).values / self.dtc.get_depth()

    def tree_depth_pruned(self):
        return self.X.apply(lambda x: self.dtc_pruned.decision_path([x]).sum() - 1,
                            axis=1, raw=True).values / self.dtc_pruned.get_depth()

    def class_likeliood(self):
        n = len(self.y.unique())
        priors = np.ones((n,)) / n

        nb = GaussianNB(priors=priors)
        nb.fit(self.X, self.y)

        prob = nb.predict_proba(self.X)
        labels = self.y.values

        CL = [prob[i, np.argwhere(nb.classes_ == labels[i])[0][0]] for i in range(0, len(labels))]

        return 1 - np.array(CL)

    # TODO: implement for discrete variables too
    def class_likeliood_diff(self):
        n = len(self.y.unique())
        priors = np.ones((n,)) / n

        nb = GaussianNB(priors=priors)
        nb.fit(self.X, self.y)

        prob = nb.predict_proba(self.X)
        labels = self.y.values

        CLD = [prob[i, np.argwhere(nb.classes_ == labels[i])[0][0]] -
               np.max(np.delete(prob[i, :], np.argwhere(nb.classes_ == labels[i])[0][0]))
               for i in range(0, len(labels))]

        return (1 - np.array(CLD)) / 2

    def minority_value(self):
        mv_class = self.data.groupby(self.labels_col).count().iloc[:, 0]
        mv_class = mv_class.divide(mv_class.max())

        labels = self.y
        return labels.apply(lambda c: 1 - mv_class[c]).values

    def class_balance(self):
        cb_class = self.data.groupby(self.labels_col).count().iloc[:, 0]
        cb_class = cb_class.divide(self.N) - 1 / len(self.y.unique())

        labels = self.y
        return (1 - labels.apply(lambda c: cb_class[c]).values) / 1.5

    def borderline_points(self):
        y = self.y.values.copy()

        dist_matrix = self.dist_matrix_gower
        Tcsr = minimum_spanning_tree(dist_matrix)
        mst = Tcsr.toarray()
        mst = np.where(mst > 0, mst, np.inf)

        N1 = np.zeros(y.shape)
        for i in range(len(y)):
            idx = np.argwhere(np.minimum(mst[i, :], mst[:, i]) < np.inf)
            N1[i] = np.sum(y[idx[:, 0]] != y[i]) / len(y[idx[:, 0]])
        return N1

    def intra_extra_ratio(self, distance='gower'):
        y = self.y.copy()

        if distance == 'gower':
            indices = self.indices_gower
            distances = self.distances_gower
        else:
            nbrs = NearestNeighbors(n_neighbors=len(self.y), algorithm='auto', metric=distance).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        N2 = np.zeros(y.values.shape)
        for i, label in y.items():
            nn = y.loc[indices[i, :]]
            intra = nn.eq(label)
            extra = nn.ne(label)
            assert np.all(np.diff(distances[i, intra]) >= 0)
            assert np.all(np.diff(distances[i, extra]) >= 0)
            N2[i] = distances[i, intra][1] / max(distances[i, extra][0], 1e-15)
        return 1 - 1 / (N2 + 1)

    def local_set_cardinality(self, distance='gower'):
        y = self.y.copy()

        if distance == 'gower':
            indices = self.indices_gower
        else:
            nbrs = NearestNeighbors(n_neighbors=len(self.y), algorithm='auto', metric=distance).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        LSC = np.zeros(y.values.shape)
        n_class = y.value_counts()
        for i, label in y.items():
            nn = y.loc[indices[i, :]].values
            LSC[i] = (np.argmax(nn != label) - 1) / (n_class[label] - 1)
        return 1 - LSC

    def ls_radius(self, distance='gower'):
        y = self.y.copy()

        if distance == 'gower':
            indices = self.indices_gower
            distances = self.distances_gower
        else:
            nbrs = NearestNeighbors(n_neighbors=len(self.y), algorithm='auto', metric=distance).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        LSR = np.zeros(y.values.shape)
        for i, label in y.items():
            nn = y.loc[indices[i, :]].values
            aux = (nn == label)[::-1]
            ind_nn_max = len(aux) - np.argmax(aux) - 1
            LSR[i] = min(1.0, distances[i, np.argmax(nn != label)] / distances[i, ind_nn_max])
        return 1 - LSR

    def harmfulness(self, distance='gower'):
        y = self.y.values.copy()

        if distance == 'gower':
            indices = self.indices_gower
        else:
            nbrs = NearestNeighbors(n_neighbors=len(self.y), algorithm='auto', metric=distance).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        ne_pos = np.argmax(y[indices] != y[:, None], axis=1)
        ne = indices[np.arange(len(indices)), ne_pos]
        n_class = self.y.value_counts()

        H = np.sum(np.reshape(np.arange(self.N), (self.N, -1)) == ne, axis=1)
        return H / n_class[self.y].values

    def usefulness(self, distance='gower'):
        y = self.y.values.copy()

        if distance == 'gower':
            indices = self.indices_gower
        else:
            nbrs = NearestNeighbors(n_neighbors=len(self.y), algorithm='auto', metric=distance).fit(self.X)
            distances, indices = nbrs.kneighbors(self.X)

        ne_pos = np.argmax(y[indices] != y[:, None], axis=1)
        n_class = self.y.value_counts()

        u = np.zeros(y.shape)
        for i in range(self.N):
            ls = indices[i, 1:ne_pos[i]]
            u[ls] += 1
        return 1 - (u / n_class[self.y].values)

    def f1(self):
        df = self.data.copy()
        features = self.X.columns.to_list()
        n_features = len(features)
        classes = self.y.unique().tolist()

        F1 = pd.Series(0, index=df.index)
        for p in itertools.combinations(classes, 2):
            sub_df = df[(self.y == p[0]) | (self.y == p[1])]
            indicator = pd.Series(0, index=sub_df.index)
            for f in features:
                m1 = maxmin(sub_df[f].values, sub_df[self.labels_col].values)
                m2 = minmax(sub_df[f].values, sub_df[self.labels_col].values)
                indicator += sub_df[f].between(m1, m2, inclusive=False) * 1
            F1 = F1.add(indicator.divide(n_features), fill_value=0)

        F1 = F1.divide(len(classes) - 1)
        # assert F1.max() <= 1.0
        return F1.values

    def do_t(self):
        df = self.data.copy()
        features = self.X.columns.to_list()
        classes = self.y.unique().tolist()

        dot = pd.DataFrame(0, columns=self.X.columns, index=self.X.index)
        for p in itertools.combinations(classes, 2):
            sub_df = df[(self.y == p[0]) | (self.y == p[1])]
            for f in features:
                m1 = maxmin(sub_df[f].values, sub_df[self.labels_col].values)
                m2 = minmax(sub_df[f].values, sub_df[self.labels_col].values)
                do = (m2 - sub_df[f]) / (m2 - m1)
                dot[f] = dot[f].add(1 / (1 + abs(0.5 - do)), fill_value=0)

        k = len(classes)
        dot = dot.div(k * (k - 1) / 2)
        return dot

    def f2(self):
        if self.dot is None:
            self.dot = self.do_t()
        return self.dot.min(axis=1).values

    def f3(self):
        if self.dot is None:
            self.dot = self.do_t()
        return self.dot.mean(axis=1).values

    def f4(self):
        if self.dot is None:
            self.dot = self.do_t()
        return self.dot.max(axis=1).values
