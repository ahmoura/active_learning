import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt

from copy import deepcopy

from sklearn.model_selection import ShuffleSplit, train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from timeit import default_timer as timer

from tqdm import tqdm

from scipy.io import arff