import copy
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from util.load import load
from util.torch import FeatureExtractor


##
#  Calculates system AUC+EER and provides FPR and TPR lists
#  used for creating the ROCAUC plot.
#
def evaluate(data):
    labels = [int(e) for e in data['labels']]
    scores = [float(e) for e in data['scores']]
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    auc = metrics.roc_auc_score(np.array(labels), np.array(scores))
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return (tpr, fpr, auc, eer)

##
#  Train and evaluate the system.
#
def train_evaluate(params):
    raw, feat_ext, same_day, unreg, steps, identification = params

    train_user_dfs, train_neg_df, test_user_dfs, test_neg_df = \
        load(raw, same_day, unreg, feat_ext)

    # train_neg_df.to_csv('torch_session0.csv', sep='\t')

    y_train_neg = train_neg_df[train_neg_df.columns[-1]].values
    X_train_neg = train_neg_df.drop(train_neg_df.columns[-1], axis=1).values

    y_test_neg = test_neg_df[test_neg_df.columns[-1]].values
    X_test_neg = test_neg_df.drop(test_neg_df.columns[-1], axis=1).values

    # Scores used for micro-averaging
    auc_list = []
    eer_list = []

    system_scores = pd.DataFrame({'labels': [], 'scores': []})

    if identification:
        model = SVC(kernel='rbf', gamma='auto', C=100)

        X_train = copy.deepcopy(X_train_neg)
        y_train = copy.deepcopy(y_train_neg)

        X_test = copy.deepcopy(X_test_neg)
        y_test = copy.deepcopy(y_test_neg)

        if steps > 1:
            y_test_neg = np.zeros(len(y_test_neg[:-(steps - 1)]))
        else:
            y_test_neg = np.zeros(len(y_test_neg))

        for idx in range(len(train_user_dfs)):
            train_df = train_user_dfs[idx]
            test_df = test_user_dfs[idx]

            # Compile data for training
            y_pos = train_df[train_df.columns[-1]].values
            X_pos = train_df.drop(train_df.columns[-1], axis=1).values

            X_train = np.concatenate((X_train, X_pos), axis=0)
            y_train = np.concatenate((y_train, y_pos), axis=0)

            # Compile data for testing
            y_pos = test_df[test_df.columns[-1]].values
            X_pos = test_df.drop(test_df.columns[-1], axis=1).values

            X_test = np.concatenate((X_test, X_pos), axis=0)
            y_test = np.concatenate((y_test, y_pos), axis=0)

        model.fit(X_train, y_train)
        return metrics.accuracy_score(y_test, model.predict(X_test))
    else:
        models = [SVC(kernel='rbf', gamma='auto', C=100)] * len(train_user_dfs)

        if steps > 1:
            y_test_neg = np.zeros(len(y_test_neg[:-(steps - 1)]))
        else:
            y_test_neg = np.zeros(len(y_test_neg))

        for idx in range(len(train_user_dfs)):
            train_df = train_user_dfs[idx]
            test_df = test_user_dfs[idx]
            model = models[idx]

            # Compile data for training
            y_pos = train_df[train_df.columns[-1]].values
            X_pos = train_df.drop(train_df.columns[-1], axis=1).values

            X_train = np.concatenate((X_pos, X_train_neg), axis=0)
            y_train = np.concatenate((y_pos, y_train_neg), axis=0)

            # Compile data for testing
            y_pos = test_df[test_df.columns[-1]].values
            X_pos = test_df.drop(test_df.columns[-1], axis=1).values

            model.fit(X_train, y_train)

            # Evaluate for consecutive steps
            y_test_pos = np.ones(len(y_pos) - steps + 1)
            y_scores_pos = model.decision_function(X_pos).tolist()
            if steps > 1:
                y_scores_pos = [np.mean(y_scores_pos[i:(i + steps)])
                    for i in range(len(y_scores_pos) - steps + 1)]

            y_scores_neg = model.decision_function(X_test_neg).tolist()
            if steps > 1:
                y_scores_neg = [np.mean(y_scores_neg[i:(i + steps)])
                    for i in range(len(y_scores_neg) - steps + 1)]

            scores = np.concatenate((y_scores_pos, y_scores_neg), axis=0)
            labels = np.concatenate((y_test_pos, y_test_neg), axis=0)

            try:
                auc = metrics.roc_auc_score(labels, scores)
                auc_list.append(auc)
            except ValueError:
                print('Exception at user', idx)

            fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            eer_list.append(eer)
            
            system_scores = system_scores.append(pd.DataFrame(
                data=np.c_[labels, scores], columns=['labels', 'scores']))

        m_auc  = np.mean(auc_list)
        sd_auc = np.std(auc_list)
        print("User AUC (mean, stdev): {}, {}".format(m_auc, sd_auc))
        m_eer = np.mean(eer_list)
        sd_eer = np.std(eer_list)
        print("User EER (mean, stdev): {}, {}".format(m_eer, sd_eer))

        return system_scores
