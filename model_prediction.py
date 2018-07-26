## Import libraries

# ---- Import basic library
import sys
import json
from collections import Counter  
import collections
import itertools
from itertools import chain
import pydot
import re

# ---- Import numpy library
import numpy
import numpy as np
from numpy import math
from numpy import argmax

# ---- Import Scipy library
from scipy.sparse import csr_matrix

# ---- Import pandas library
import pandas as pd

# ---- Import seaborn
import seaborn as sns

# ---- Import TensorFlow library
import tensorflow as tf

# ---- Import from matplotlib library
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---- Import scikit-learn library
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
#introduced recentl in version 0.20 release.
#from sklearn.preprocessing import CategoricalEncoder  

# ---- Import Keras deep neural network library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1, l2 #,WeightRegularizer
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K

# ---- Import Imbalanced library
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline 
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from imblearn.ensemble import BalancedBaggingClassifier 
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import classification_report_imbalanced

# ---- Import Scikit-Learn Optimizer library
from skopt import gp_minimize, forest_minimize
from skopt.plots import plot_evaluations, plot_convergence, plot_objective
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# ---- Import MLXtend library
#from mlxtend.evaluate import confusion_matrix
#from mlxtend.plotting import plot_confusion_matrix

# ---- Import neural network modelling
from neural_network.models import DeepModel

# ---- Import optimization for hyper-parameter
from neural_network.optimization import SkOptObjective

# ---- Import plotter
from visualization.plotter import Plotter

# ---- Import Warning library
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ---- Fix random seed for reproducibility
seed = 42
numpy.random.seed(seed)
np.set_printoptions(precision=2)

# ---- Import model, dataloader, and metrics  module
from visualization import *
from models import *
from dataloader import *
from metrics import *




# ---- Declare plotter 
plotter = Plotter()

# ---- Set of available input files
input_filenames = ['history.180412.json', 'history.180223.json','history.171102.json', 'poolview_totals.json']
data = pd.read_json('data/'+input_filenames[0], orient='index')

# ---- Rest index
data_index_reset = data.reset_index()


# ---- Add column with list of exit code for good/bad sites
data_index_reset['errors_good_sites_exit_codes'] = data_index_reset['errors'].apply(lambda x: x['good_sites'].keys() if len(x['good_sites'].keys()) != 0 else ['0'])

data_index_reset['errors_bad_sites_exit_codes'] = data_index_reset['errors'].apply(lambda x: x['bad_sites'].keys() if len(x['bad_sites'].keys()) != 0 else ['0'])


# ---- Add column of dictionary with a list of good/bad sites
data_index_reset['errors_good_sites_dict'] = data_index_reset['errors'].apply(lambda x: x['good_sites'].values() if len(x['good_sites'].values()) !=0 else [{'NA': 0}])

data_index_reset['errors_bad_sites_dict'] = data_index_reset['errors'].apply(lambda x: x['bad_sites'].values() if len(x['bad_sites'].values()) !=0 else [{'NA': 0}])


# ---- Add column of list of good/bad sites
data_index_reset['errors_good_sites_list'] = data_index_reset['errors_good_sites_dict'].apply(lambda x:  list_of_sites(x))

data_index_reset['errors_bad_sites_list'] = data_index_reset['errors_bad_sites_dict'].apply(lambda x:  list_of_sites(x))


# ---- Exit code based on good_sites
list2d = data_index_reset['errors_good_sites_exit_codes'].tolist()
good_sites_exit_codes = sorted(set(list(itertools.chain.from_iterable(list2d))),key=int)
good_sites_exit_codes = [str(x) for x in good_sites_exit_codes]

# ---- Extract exit code based on bad_sites
list2d = data_index_reset['errors_bad_sites_exit_codes'].tolist()
bad_sites_exit_codes = sorted(set(list(itertools.chain.from_iterable(list2d))),key=int)
bad_sites_exit_codes = [str(x) for x in bad_sites_exit_codes]

# ---- Extract site names from good sites 
list2d_step1 = data_index_reset['errors_good_sites_list'].tolist()
list2d_step2 = list(itertools.chain.from_iterable(list2d_step1))
good_site_names = sorted(set(list(itertools.chain.from_iterable(list2d_step2))))
good_site_names = [str(x) for x in good_site_names]

# ---- Extract site names from bad sites
list2d_step1 = data_index_reset['errors_bad_sites_list'].tolist()
list2d_step2 = list(itertools.chain.from_iterable(list2d_step1))
bad_site_names = sorted(set(list(itertools.chain.from_iterable(list2d_step2))))
bad_site_names = [str(x) for x in bad_site_names]


# ---- Build good/bad site features
data_index_reset['table_good_sites'] = data_index_reset['errors'].apply(lambda x: 
                                                                        build_table(x['good_sites'], 
                                                                                    good_site_names, 
                                                                                    good_sites_exit_codes))


data_index_reset['table_bad_sites'] = data_index_reset['errors'].apply(lambda x: 
                                                                       build_table(x['bad_sites'], 
                                                                                   bad_site_names, 
                                                                                   bad_sites_exit_codes))


# ---- Flatten good/bad site features
data_index_reset['table_good_sites_flatten'] = data_index_reset['table_good_sites'].apply(lambda x: 
                                                                                          build_table_flatten(x))


data_index_reset['table_bad_sites_flatten'] = data_index_reset['table_bad_sites'].apply(lambda x: 
                                                                                        build_table_flatten(x))


# ---- Combined flatten good/bad site features
data_index_reset['table_combined_sites_flatten'] =  data_index_reset.apply(lambda x:
                                                                           combine_features(x,
                                                                                            'table_good_sites_flatten',
                                                                                            'table_bad_sites_flatten'),
                                                                           axis=1)



# ---- Add column with splitting categorical levels
data_index_reset['splitting'] = data_index_reset['parameters'].apply(lambda x: 
                                                                     splitting_fnc(x,
                                                                                'splitting'))


splitting_categories = sorted(list(set(data_index_reset['splitting'])))

# ---- Encode splitting categorical levels
data_index_reset['splitting_encoded'] = data_index_reset['splitting'].astype(pd.api.types.CategoricalDtype(categories = 
                                                                                                           splitting_categories)).cat.codes



# ---- Definite feature column called 'action' which is take as the target variable
data_index_reset['action'] = data_index_reset['parameters'].apply(lambda x: x['action'])


action_categories = sorted(list(set(data_index_reset['action'])))

# ---- Encode action categorical levels
data_index_reset['action_encoded'] =  data_index_reset['action'].astype(pd.api.types.CategoricalDtype(categories = 
                                                                                                      action_categories)).cat.codes


data_index_reset['action_encoded'].value_counts()


# ---- Target categorical levels
data_index_reset['target_label'] = data_index_reset.apply(lambda x:
                                                          merge_labels(x,
                                                                       ['action',
                                                                        'splitting']),
                                                          axis=1)



target_categories = sorted(list(set(data_index_reset['target_label'])))


# ---- Encode target categorical levels
data_index_reset['target_encoded'] = data_index_reset['target_label'].astype(pd.api.types.CategoricalDtype(categories =
                                                                                                           target_categories)).cat.codes

data_index_reset['target_encoded'].value_counts()


labels = list(set(data_index_reset['target_encoded'].tolist()))


# ---- Add column with xrootd levels
data_index_reset['xrootd'] = data_index_reset['parameters'].apply(lambda x: 
                                                                  xrootd_fnc(x, 'xrootd'))


# ---- Encode xrootd levels (not in particular use at the moment)
data_index_reset['xrootd_encoded'] = data_index_reset['xrootd'].astype(pd.api.types.CategoricalDtype(categories = ["NaN", "disabled", "enabled"])).cat.codes


data_index_reset['xrootd'].value_counts()

data_index_reset.head()

# ---- Plot the frequency count per target category
plot_class_count(data_index_reset, 'target_label')


# ---- Plot the total number of instances for each exit code through all sites per workflow
exit_code_counts(data_index_reset, 'table_good_sites', title='good')


# ---- Plot the total number of instances for each exit code through all sites per workflow
exit_code_counts(data_index_reset, 'table_bad_sites', title='bad')


# ---- Definte new feature column containing campain info
#data_index_reset['campaign'] = data_index_reset[['index']].apply(lambda x: extract_campaign(x), axis=1)


# ---- Setup data for training and evaluation
#  features           
X = np.array(data_index_reset['table_combined_sites_flatten'].tolist())

# target
y = np.array(data_index_reset['target_encoded'].tolist())

# ---- Perform train test split 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    shuffle=True,
                                                    random_state=seed)


# some time later...

# Loading whole models (architecture + weights + optimizer state)
from keras.models import load_model


# returns a compiled model
# identical to the previous one
model = load_model('model_weights/nn_model.h5')


# evaluate test dataset
score = model.evaluate(np.array(X_test), np.array(y_test), batch_size=128)
print(np.mean(score))

# make prediction on test dataset
print(model.predict(np.array(X_test)))
