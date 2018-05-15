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
