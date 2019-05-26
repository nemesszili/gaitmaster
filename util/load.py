from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from util.torch import FeatureExtractor

from util.const import *


class Extractor():
    def __init__(self):
        self.data = None

extractor = Extractor()

##
#
#
def scale_df(df):
    X = df.drop([df.columns[-1]], axis=1)
    df[df.columns[:-1]] = scale(X)
    return df

##
#  Reads the content of CSV file containing 59 feature data.
#
def load_feat(csv):
    df = pd.read_csv(Path(PATH).joinpath(Path(csv)), header=None)
    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    df[df.columns[-1]] = y
    
    return df

##
#  Reads the content of CSV file containing raw step cycles.
#
def load_raw(csv):
    df = pd.read_csv(Path(PATH).joinpath(Path(csv)), sep='\t')
    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    df[df.columns[-1]] = y
    
    return df

##
#  Compiles the name of a 59 feature CSV based on the session.
#
def get_feat_csv(session):
    return 'zju_gaitaccel_session_' + str(session) + '_128.csv'

##
#  Compiles the name of a raw CSV based on the session.
#
def get_raw_csv(session):
    return 'zju_raw_session_' + str(session) + '_128.csv'

##
#  Filters dataframe based on label.
#
def filter_df(df, label):
    return df.loc[df[df.columns[-1]] == label]

##
#  Loads data used for training autoencoders.
#
def load_auto(raw):
    if raw:
        df_s0 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(0))), sep='\t')
        df_s1 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(1))), sep='\t')
        df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(2))), sep='\t')
    else:
        df_s0 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(0))), header=None)
        df_s1 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(1))), header=None)
        df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(2))), header=None)

    return pd.concat([df_s0, df_s1, df_s2])

##
#  Prepares the data for the measurement.
#
#  Steps:
#   1.   Loads as many dataframes as user classes we want to measure.
#   2.   Samples negative data for training
#   3.1. Same day: use the first half of every user dataframe for training 
#        and the second half for testing. Take negative training samples from 
#        either the rest or session 0. Negative test samples also from session 0.
#   3.2. Cross day: Use the first half of every user dataframe from seesion 2
#        for testing and use negative samples from either the rest or session 0.
#
def load(raw, same_day, unreg, feat_ext):
    if extractor.data is None:
        if raw:
            df_s0 = load_raw(get_raw_csv(0))
            df_s1 = load_raw(get_raw_csv(1))
            df_s2 = load_raw(get_raw_csv(2))
        else:
            df_s0 = load_feat(get_feat_csv(0))
            df_s1 = load_feat(get_feat_csv(1))
            df_s2 = load_feat(get_feat_csv(2))

        if feat_ext is not None:
            if raw:
                extr = FeatureExtractor(load_auto(raw), feat_ext, [384, 192, 64])
            else:
                extr = FeatureExtractor(load_auto(raw), feat_ext, [59, 32, 16])

            df_s0 = extr.to_latent(df_s0)
            df_s1 = extr.to_latent(df_s1)
            df_s2 = extr.to_latent(df_s2)

        df_s0 = scale_df(df_s0)
        df_s1 = scale_df(df_s1)
        df_s2 = scale_df(df_s2)
    else:
        return extractor.data
        
    # Create user dataframes
    train_user_dfs = list(map(lambda c: filter_df(df_s1, c), USER_RANGE))
    pos_size = int(np.mean([len(df) for df in train_user_dfs]))
    
    train_neg_dfs = list(map(lambda c: filter_df(df_s0, c), S0_TRAIN_USER_RANGE))
    
    # Halve all user dataframes; use the first halves for training,
    # second halves for testing
    new_train = []
    test_user_dfs = []
    for df in train_user_dfs:
        df1, df2 = np.array_split(df, 2)
        new_train.append(df1)
        test_user_dfs.append(df2)
    train_user_dfs = new_train

    new_neg_train = []
    test_neg_dfs = []
    for df in train_neg_dfs:
        df1, df2 = np.array_split(df, 2)
        new_neg_train.append(df1)
        test_neg_dfs.append(df2)
    train_neg_df = pd.concat(new_neg_train)

    if unreg:
        test_neg_dfs = list(map(lambda c: filter_df(df_s0, c), S0_TEST_USER_RANGE))
        test_neg_dfs = list(map(lambda df: df.head(int(len(df)/2)), test_neg_dfs))

    if not same_day:
        test_user_dfs = list(map(lambda c: filter_df(df_s2, c), USER_RANGE))
        test_user_dfs = list(map(lambda df: df.head(int(len(df)/2)), test_user_dfs))

    test_neg_df = pd.concat(test_neg_dfs)

    train_neg_df[train_neg_df.columns[-1]] = 0
    test_neg_df[test_neg_df.columns[-1]] = 0

    extractor.data = (train_user_dfs, train_neg_df,
                      test_user_dfs, test_neg_df)

    return train_user_dfs, train_neg_df, test_user_dfs, test_neg_df
