#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:39:28 2017

@author: carlos
"""

import numpy as np
import pandas
import time
import matplotlib.pyplot as plt
from plot_learning_curve import plot_learning_curve
from plot_validation_curve import plot_validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


# function that loads train.csv, ignores id column, resamples data to 
# frac ratio and creates time, hour and day of week columns. 
# has an option to save results to file

def parseInitialData(frac=0.1, usecols=range(1,24), save_result=False):
    print "Parsing file... "
    start_time = time.time()
    data = pandas.read_csv("train.csv", nrows=None, usecols=usecols)
    print "Time parsing file = ", time.time() - start_time
    
    print "Sampling {:.1f}% of data...".format(frac*100.0) 
    data = data.sample(frac=frac)
    
    print "Converting time..."
    data["time"]=pandas.to_datetime("20"+data.hour.astype(str),
                format="%Y%m%d%H")
    print "Creating hour..."
    data["hour"]=data["time"].apply(lambda x: x.hour)
    print "Creating day of week..."
    data["day"]=data["time"].apply(lambda x: x.dayofweek)
    
    if save_result == True:
        data.to_csv("reduced_"+str(frac)+".csv", index=False)
    
    return data
    
""" 
function that runs a category number analysis as well as computes
the variance of the click rate in each feature
""" 
def analyzeData(data):
    print "# categories / data size: "
    
    feat_number_categ = list()
    ctr_var = []

    # compute number of categories and variance of click rate for each feature
    for k in data.keys():
        group = data.groupby(k)
        
        #print k, " = {:.1f}%".format(100.0*len(group)/len(data)), "% (", \
        #        len(group), ")"
        feat_number_categ.append(len(group))
        
        group = data.groupby(k)
        ctr_var.append((group.click.sum()/group.size()).var())
   
    # plot results
    plt.figure()
    plt.bar(range(2,len(data.keys())+1), feat_number_categ[1:], align="center")
    plt.xticks(range(2,len(data.keys())+1), list(data.keys()[1:]), 
               rotation='vertical')
    plt.title("Number of categories by feature")
    plt.savefig("num_categ_excluding")
    print "Total categories = ", sum(feat_number_categ)
    
    plt.figure()
    plt.bar(range(2,len(data.keys())+1), ctr_var[1:], align="center")
    plt.xticks(range(2,len(data.keys())+1), list(data.keys()[1:]), 
               rotation='vertical')
    plt.title("Variance of category click-rate by feature")
    
    return plt

""" 
function that resamples X,y datasets to a specified ratio of positive
examples
""" 
def resampling(X,y,ratio):
    pos_max = sum(y)
    neg_max = len(y)-sum(y)
    
    n_pos = ratio*(pos_max+neg_max)
    n_neg = (1-ratio)*(pos_max+neg_max)
    if n_neg > neg_max:
        n_neg = int(neg_max)
        n_pos = int(ratio*(pos_max+n_neg))
    if n_pos > pos_max:
        n_pos = int(pos_max)
        n_neg = int((1-ratio)*(neg_max+n_pos))

    Xpos = X[y==1][0:n_pos]
    ypos = y[y==1][0:n_pos]
    Xneg = X[y==0][0:n_neg]
    yneg = y[y==0][0:n_neg]

    X = np.concatenate((Xpos.toarray(),Xneg.toarray()))
    y = np.concatenate((ypos,yneg))
    return X, y

# function that encodes categorical features into one-hot encoding
def preProcess(data, label_col, non_numerical=[]):    
    # feature C20 has some -1 values that are probably not categorized rows. \
    # for now just treat unkown as a category of its own
    if "C20" in data.keys():
        data.loc[:,"C20"] = data["C20"]+1

    # encode non numerical features into numerical ones
    non_numerical = data.keys()[data.dtypes != "int64"]
    encoders = []
    for feature in non_numerical:
        fenc = preprocessing.LabelEncoder()
        fenc.fit(data[feature])
        encoders.append(fenc)
        data.loc[:,feature] = fenc.transform(data[feature])
    
    # create X matrix with just the features and y vector with class label
    X = np.array(data[data.keys().difference(["click"])])
    y = np.array(data["click"])

    # encode feature matrix X with one-hot encoding
    ohenc = preprocessing.OneHotEncoder()
    X = ohenc.fit_transform(X)
    
    return X, y, encoders
 
# add benchmark to learning curve pot and adjust y axis
def addBenchToPlot(plt):
    #bench_all05 = log_loss([1,0], [0.5,0.5])
    bench_allctr = log_loss(np.concatenate((np.ones(1698),np.zeros(8302))), 
                        1698.0/10000*np.ones(10000))
    
    max_x = int(plt.axis()[1])
    min_x = int(plt.axis()[0])

    plt.plot(range(min_x, max_x), -bench_allctr*np.ones(max_x), '-', color="b",
             label="CTR Benchmark")
    plt.legend()
    axes = plt.axes()
    axes.set_ylim(axes.get_ylim()[0], axes.get_ylim()[1]+0.01)
        
        
using_cols=[
 'click',
 'hour',
 'C1',
 'banner_pos',
 'site_id',
 'site_domain',
 'site_category',
 'app_id',
 'app_domain',
 'app_category',
 'device_model',
 'device_type',
 'device_conn_type',
 'C14',
 'C15',
 'C16',
 'C17',
 'C18',
 'C19',
 'C20',
 'C21']

print "Loading data..."

data = parseInitialData(frac=0.001, usecols=using_cols)

# load reduced data
#start_time = time.time()
#data = pandas.read_csv("reduced_0.05.csv", nrows=0.4e6, 
#                        usecols=using_cols+["day"])
#print "Time loading file = ", time.time() - start_time

analyzeData(data)

print "Preprocessing data..."

X,y,enc = preProcess(data, ["click"])


print "Splitting data set..."
X_train, X_test, y_train, y_test = train_test_split(X, \
                            y, test_size=0.2, random_state=42)


print "Training model..."

logReg = LogisticRegression(max_iter=500,C=0.3)

start_time = time.time()
logReg.fit(X_train, y_train)
print "Time took training = ", time.time() - start_time


print "Log loss regression (test/train) : {:.5f}/{:.5f}".format( \
        log_loss(y_test, logReg.predict_proba(X_test)), \
        log_loss(y_train, logReg.predict_proba(X_train)))

print "Log loss p(click) = 0.5 : {:.5f}".format( \
    log_loss(y_test, 0.5*np.ones(len(y_test))))
print "Log loss p(click) = {:.5f} : {:.5f}".format(1.0*y.sum()/len(y),
    log_loss(y_test, 1.0*y.sum()/len(y)*np.ones(len(y_test))))

# scorer for log loss
logl_sc = make_scorer(log_loss,needs_proba=True,greater_is_better=False)

# cross validation splitter (5x 70-30)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

#print "Cross-val score = {:.5f}".format(\
#    cross_val_score(logReg, X_train, y_train, scoring=logl_sc, cv=cv).mean())

# plot learning curve
lc = plot_learning_curve(logReg, "LogReg", X_train, y_train, 
                         score=logl_sc, cv=cv, n_jobs=4)


# add CTR benchmark to learning curve plot
addBenchToPlot(lc)

# plot regularization validation curve
plot_validation_curve(logReg, X_train, y_train, title="Regularization",
                      ylim=None, cv=cv, score=logl_sc, n_jobs=4,
                      param_range = np.logspace(-2,0,5))