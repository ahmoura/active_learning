import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
from sklearn import preprocessing

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

from timeit import default_timer as timer

from scipy.io import arff

from tqdm import tqdm, trange