"""
==========================
Plotting Validation Curves
==========================

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.
"""
#print(__doc__)

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator, X, y, ylim=None, title="Validation", 
                          cv=None, score=None, n_jobs=1, 
                          param_range = np.logspace(-2,0,5)):
    
    train_scores, test_scores = validation_curve(estimator, X, y,
                                param_name="C", param_range=param_range,
                                cv=cv, scoring=score, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Regularization parameter")
    plt.ylabel("Score")
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    lw = 2
    plt.grid()
    plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    return plt