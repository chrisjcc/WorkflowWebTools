## Import libraries

# ---- Import basic library
import sys
import json
from collections import Counter  
import collections
import itertools
from itertools import chain
import pydot
import graphviz
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

# ---- Import from holoviews
import holoviews as hv
hv.extension('bokeh')

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
from sklearn.preprocessing import CategoricalEncoder  

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
from skopt.plots import plot_evaluations, plot_convergence, plot_objective#, plot_histogram
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# ---- Import MLXtend library
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

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
from models import *
from dataloader import *
from metrics import *




# ---- Declare plotter 
plotter = Plotter()

# ---- Set of available input files
input_filenames = ['history.180412.json', 'history.180223.json','history.171102.json', 'poolview_totals.json']
data = pd.read_json(input_filenames[0], orient='index')

# ---- Check table, shape, size, and dimension
print(data.head())
print(type(data))
print(data.shape)
print(data.size) # size = shape*ndim
print(data.ndim)

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

print(len(good_sites_exit_codes))
print(type(good_sites_exit_codes))
print(good_sites_exit_codes)

# ---- Extract exit code based on bad_sites
list2d = data_index_reset['errors_bad_sites_exit_codes'].tolist()
bad_sites_exit_codes = sorted(set(list(itertools.chain.from_iterable(list2d))),key=int)
bad_sites_exit_codes = [str(x) for x in bad_sites_exit_codes]

print(len(bad_sites_exit_codes))
print(type(bad_sites_exit_codes))
print(bad_sites_exit_codes)


# ---- Extract site names from good sites 
list2d_step1 = data_index_reset['errors_good_sites_list'].tolist()
list2d_step2 = list(itertools.chain.from_iterable(list2d_step1))
good_site_names = sorted(set(list(itertools.chain.from_iterable(list2d_step2))))
good_site_names = [str(x) for x in good_site_names]

print(len(good_sites_exit_codes))
print(len(good_site_names))
print(good_site_names)


# ---- Extract site names from bad sites
list2d_step1 = data_index_reset['errors_bad_sites_list'].tolist()
list2d_step2 = list(itertools.chain.from_iterable(list2d_step1))
bad_site_names = sorted(set(list(itertools.chain.from_iterable(list2d_step2))))
bad_site_names = [str(x) for x in bad_site_names]

print(len(bad_sites_exit_codes))
print(len(bad_site_names))
print(bad_site_names)


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
data_index_reset['dummy_good_sites_flatten'] = data_index_reset['dummy_good_sites'].apply(lambda x: 
                                                                                          build_table_flatten(x))


data_index_reset['dummy_bad_sites_flatten'] = data_index_reset['dummy_bad_sites'].apply(lambda x: 
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
print(splitting_categories)


# ---- Encode splitting categorical levels
data_index_reset['splitting_encoded'] = data_index_reset['splitting'].astype(pd.api.types.CategoricalDtype(categories = 
                                                                                                           splitting_categories)).cat.codes



# ---- Definite feature column called 'action' which is take as the target variable
data_index_reset['action'] = data_index_reset['parameters'].apply(lambda x: x['action'])


action_categories = sorted(list(set(data_index_reset['action'])))
print(set(data_index_reset['action']))
print(set(data_index_reset['splitting']))

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



print(set(data_index_reset['target_label']))
print(len(set(data_index_reset['target_label'])))
print(data_index_reset['target_label'].value_counts())
target_categories = sorted(list(set(data_index_reset['target_label'])))
print(target_categories)


# ---- Encode target categorical levels
data_index_reset['target_encoded'] = data_index_reset['target_label'].astype(pd.api.types.CategoricalDtype(categories =
                                                                                                           target_categories)).cat.codes

data_index_reset['target_encoded'].value_counts()


labels = list(set(data_index_reset['target_encoded'].tolist()))
print(labels)


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



data_index_reset.head()


# ---- Definte new feature column containing campain info
data_index_reset['campaign'] = data_index_reset[['index']].apply(lambda x: extract_campaign(x), axis=1)


# ---- Loop through all those campaigns that have exit code in the 8000 range in good site info
for campaign in campaign_list:
    print(campaign)
    plot_campaign_exit_codes(campaign, "errors_good_sites_exit_codes")

# ---- Loop through all those campaigns that have exit code in the 8000 range in bad site info
for campaign in campaign_list:
    print(campaign)
    plot_campaign_exit_codes(campaign, "errors_bad_sites_exit_codes")



# ---- Setup data for training and evaluation
#  features
X = data_index_reset['dummy_combined_sites_flatten'].tolist()

# target
y = data_index_reset['target_encoded'].tolist()

# ---- Perform train test split 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    shuffle=True,
                                                    random_state=seed)


# ---- Preprocessing using 0-1 scaling byremoving the mean and scaling to unit variance 
scaler = RobustScaler()

# ---- Use early stopping on training when the validation loss isn't decreasing anymore
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


## Configure neural network architecture

# ---- Set configuration
input_dim    = np.array(X).shape[1]
n_classes    = len(set(y))
nlayers      = 3  
nneurons     = 51
l2_norm      = 0.0014677547170664112
dropout_rate = 0.014655354118727714
loss         = 'sparse_categorical_crossentropy'

default_parameters = [3, 51, 0.0014677547170664112, 0.014655354118727714]
print('input_dim', input_dim)
print('n_classes', n_classes)

# ---- Create model for use in scikit-learn
pipe = {
    'kerasclassifier':  make_pipeline(scaler,
                                      KerasClassifier(build_fn=create_model,
                                                      input_dim=input_dim,
                                                      n_classes=n_classes,
                                                      nlayers=nlayers,
                                                      nneurons=nneurons,
                                                      dropout_rate=dropout_rate,
                                                      l2_norm=l2_norm,
                                                      loss=loss,
                                                      batch_size=256, 
                                                      epochs=35,
                                                      #validation_split=0.20,
                                                      #callbacks=[early_stopping],
                                                      verbose=1))
}



# ---- Declare model instance
model = pipe['kerasclassifier']


# ---- Fit (i.e. train) model based on default setting
model.fit(X_train, y_train)


## Model performance

# ---- Predictions based on testing data
y_pred = model.predict(X_test)

# ---- Probability predictions based on testing data
y_prob = model.predict_proba(X_test)

# ---- Defined as the negative log-likelihood of the true labels and predictions.
print('log_loss: ', log_loss(y_test, y_prob, labels=labels))

# ---- Evaluate test prediction according to accuracy
print('accuracy: ', accuracy_score(y_test, y_pred))

# ---- Evaluate traing prediction according to recall
# The recall is the ratio tp/(tp + fn) where tp is the number of true positives 
# and fn the number of false negatives. The recall is intuitively the ability 
# of the classifier to find all the positive samples.
# number of correctly predicted "positives" divided by the total number of "positives".
print('recall: ', recall_score(y_test, y_pred, average='weighted'))


# ---- Evaluate traing prediction according to precision
# The precision is the ratio tp / (tp + fp) where tp is the number of true positives 
# and fp the number of false positives. The precision is intuitively the ability 
# of the classifier not to label as positive a sample that is negative
print('precision: ', precision_score(y_test, y_pred, average='weighted'))


# ---- F1-score corresponds to the harmonic mean of precision and recal
print('f1-score: ', f1_score(y_test, y_pred, average='weighted'))


# ---- The geometric mean corresponds to the square root of the product of the 
# sensitivity and specificity. Combining the two metrics should account 
# for the balancing of the dataset.
# Calculate metrics for each label, and find their average, weighted by support 
# (the number of true instances for each label). This alters ‘macro’ to account 
# for label imbalance; it can result in an F-score that is not between precision and recall.
print(geometric_mean_score(y_test, y_pred, average='weighted'))


# ---- Classify and report the results
print(classification_report_imbalanced(y_test, y_pred))


# ---- Extract column names
index = list(set(y_test))
columns = [target_categories[i] for i in index]

# ---- Plot non-normalized confusion matrix
plotter.plot_confusion_matrix(y_test, y_pred,
                              normalize=False, classes=columns,
                              title='Confusion matrix, without normalization')

# ---- Plot normalized confusion matrix
plotter.plot_confusion_matrix(y_test, y_pred, classes=columns,
                              normalize=True, title='Normalized confusion matrix')


## Bayesian optimization using Gaussian Processes

# ---- Build scikit-learn scoring metric for log-loss
neg_log_loss_scoring = make_scorer(neg_log_loss, greater_is_better=False, needs_proba=True)

# ---- Build scikit-learn scoring metric for f1-score
f1_scoring = make_scorer(f1_score_objective, greater_is_better=True, needs_proba=False)

# ---- Build scikit-learn scoring metric for geometric-mean-score
geometric_mean_scoring = make_scorer(geometric_mean, greater_is_better=True, needs_proba=False)


# configuration space
space  = [
    Integer(1,   15,                       name='kerasclassifier__nlayers'),
    Integer(5,   75,                       name='kerasclassifier__nneurons'),
    Real(10**-3, 9.*10**-1, "log-uniform", name='kerasclassifier__l2_norm'),
    Real(10**-4, 10**-1,    "log-uniform", name='kerasclassifier__dropout_rate')
    #Real(10**-6, 10**-2,    "log-uniform", name='kerasclassifier__learning_rate'),
    #Categorical(categories=['relu', 'sigmoid'], name='activation')
    
]

# Note: try 10 number of neurons for lower bound and 10**-4 for l2 norm/dropout

dim_names = ['n_layers', 'n_neurons', 'l2_norm', 'dropout_rate']

# number of iterations
n_calls = 11

# K-fold stratified cross-validaiton
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

@use_named_args(space)
def objective(**params):

    model.set_params(**params)
    
    score = -np.mean(cross_val_score(model,
                                     np.array(X), np.array(y),
                                     cv=cv, scoring=f1_scoring)) 

    return score

# ---- Bayesian optimization based on Gaussian process regression search (controlling the exploration-exploitation trade-off)
estimator_gp_ei = gp_minimize(func=objective,     # the function to minimize
                              dimensions=space,         # the bounds on each dimension of the optimization space
                              acq_func="EI",            # the acquisition function ("EI", "LCB", "PI")
                              n_calls=n_calls,          # the number of evaluations of the objective function (Number of calls to func)
                              random_state=seed,        # the random seed  
                              x0=default_parameters,    # help the optimizer locate better hyper-parameters faster with default values
                              n_jobs=1)                 # the number of threads to use


print("Best score=%.4f (EI)" % estimator_gp_ei.fun)
print("""Expected Improvement (EI) best parameters:
- nlayers= %s  
- nneurons= %s
- l2_norm= %s
- dropout_rate= %s""" % (str(estimator_gp_ei.x[0]), str(estimator_gp_ei.x[1]),
                         str(estimator_gp_ei.x[2]), str(estimator_gp_ei.x[3])))

# ---- Evalution
plot_evaluations(estimator_gp_ei, bins=20, 
                 dimensions=dim_names);
plt.show()

# ---- Convergence (previously looked better enquire what is going on)
plot_convergence(estimator_gp_ei);
plt.show()

# ---- Partial Dependence plots are only approximations of the modelled fitness function 
# - which in turn is only an approximation of the true fitness function in fitness
plot_objective(result=estimator_gp_ei, dimensions=dim_names);
plt.show()


# ---- Aggregated target labels
data_index_reset['target_aggrated_label'] = data_index_reset.apply(lambda x:
                                                                   merge_fnc(x,
                                                                             'action',
                                                                             'splitting'),
                                                                   axis=1)


class_names=sorted(list(set(data_index_reset['target_aggrated_label'])))
print(class_names)
n_classes=len(class_names)

data_index_reset['target_aggrated_label'].value_counts()

data_index_reset['target_aggrated_encoded'] = data_index_reset['target_aggrated_label'].astype(pd.api.types.CategoricalDtype(categories = 
                                                                      class_names)).cat.codes

# ---- Plot the class count for the aggrated class label scenerio
plot_class_count(data_index_reset, 'target_aggrated_label')


# ---- Set target 
target = data_index_reset['target_aggrated_encoded'].tolist()


# ---- Perform train test split 70/30 split
train, test, target_train, target_test = train_test_split(X,
                                                          target,
                                                          test_size=0.30,
                                                          shuffle=True,
                                                          random_state=seed)

# ----- Create model for use in scikit-learn
pipe_classifiers = {
    'kerasclassifier':  make_imb_pipeline(scaler,
                                          KerasClassifier(build_fn=create_model,
                                                          input_dim=input_dim,
                                                          n_classes=n_classes,
                                                          loss=loss,
                                                          batch_size=256, 
                                                          epochs=35, 
                                                          #validation_split=0.20,
                                                          #callbacks=[early_stopping],
                                                          verbose=1))
    }


estimator = pipe_classifiers['kerasclassifier']

estimator.fit(train, target_train)

# ---- Predictions based on testing data
target_pred = estimator.predict(test)

# ---- Predictions based on testing data
target_prob = estimator.predict_proba(test)

# ---- Calculate metrics for each label, and find their average, weighted by support 
# (the number of true instances for each label). This alters ‘macro’ to account 
# for label imbalance; it can result in an F-score that is not between precision and recall.
print(geometric_mean_score(target_test, target_pred, average='weighted'))

# ---- Plot normalized confusion matrix
plotter.plot_confusion_matrix(target_test, target_pred, classes=class_names, 
                              normalize=True, title='Normalized confusion matrix')


