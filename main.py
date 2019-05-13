import dill as pickle
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from pprint import pprint

from util.process import load, load_auto
from util.torch import FeatureExtractor

# SAME DAY:
# 91.68 - 59
# 86.08 - 59_dense
# 77.49 - 59_1dcnn *
# 88.96 - raw_dense
# 83.67 - raw
# 78.42 - raw_1dcnn *
#
# CROSS DAY:
# 76.38 - 59
# 62.89 - 59_dense
# 68.29 - 59_1dcnn *
# 67.24 - raw_dense
# 65.66 - raw
# 63.14 - raw_1dcnn *

# TODO: reproducibility - set SEED
# TODO: add support for 1DCNN and Denoising autoencoder
# 1DCNN: upsample -> interpolate, generalize, refactor
# TODO: t-SNE plots
# TODO: document code
def main(raw=True, feat_ext='cnn', same_day=True):
    train_user_dfs, train_neg_df, test_user_dfs, test_neg_df = \
        load(raw=raw, split=None, same_day=same_day)

    if feat_ext:
        # Raw dense
        # extr = FeatureExtractor(load_auto(raw), feat_ext, [384, 192, 64])

        # 59 dense
        # extr = FeatureExtractor(load_auto(raw), feat_ext, [59, 32, 16])

        # Raw 1DCNN
        extr = FeatureExtractor(load_auto(raw), feat_ext, [384, 192, 96])

        # 59 1DCNN
        # extr = FeatureExtractor(load_auto(raw), feat_ext, [59, 32, 16])

        train_user_dfs = list(map(lambda df: extr.to_latent(df), train_user_dfs))
        test_user_dfs = list(map(lambda df: extr.to_latent(df), test_user_dfs))

        train_neg_df = extr.to_latent(train_neg_df)
        test_neg_df = extr.to_latent(test_neg_df)

    y_train_neg = train_neg_df[train_neg_df.columns[-1]].values
    X_train_neg = train_neg_df.drop(train_neg_df.columns[-1], axis=1).values

    y_test_neg = test_neg_df[test_neg_df.columns[-1]].values
    X_test_neg = test_neg_df.drop(test_neg_df.columns[-1], axis=1).values

    models = [SVC(kernel='rbf', gamma='auto', C=100)] * len(train_user_dfs)
    scores = []

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

        X_test = np.concatenate((X_pos, X_test_neg), axis=0)
        y_test = np.concatenate((y_pos, y_test_neg), axis=0)

        model.fit(X_train, y_train)

        fpr, tpr, _ = \
            metrics.roc_curve(y_test, model.predict(X_test), pos_label=idx+1)
        scores.append(metrics.auc(fpr, tpr))

    print(np.mean(scores))

if __name__ == '__main__':
    main()