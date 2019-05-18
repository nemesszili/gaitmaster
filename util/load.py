from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from util.const import *

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
        # df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(2))), sep='\t')
    else:
        df_s0 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(0))), header=None)
        df_s1 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(1))), header=None)
        # df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(2))), header=None)

    return pd.concat([df_s0, df_s1])

##
#  Prepares the data for the measurement. If all 153 users are used,
#  the first half (u001-u011) of session 0 is used for training and 
#  the second half (u012-u022) is used for testing.
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
def load(raw, split, same_day):
    session0 = (split is None)
    if not session0:
        POS_USER_RANGE = range(1, split + 1)
        NEG_USER_RANGE = range(split + 1, 154)
    else:
        POS_USER_RANGE = range(1, 154)

    if raw:
        df_s0 = load_raw(get_raw_csv(0))
        df_s1 = load_raw(get_raw_csv(1))
        df_s2 = load_raw(get_raw_csv(2))
    else:
        df_s0 = load_feat(get_feat_csv(0))
        df_s1 = load_feat(get_feat_csv(1))
        df_s2 = load_feat(get_feat_csv(2))

    # Create user dataframes
    train_user_dfs = list(map(lambda c: filter_df(df_s1, c), POS_USER_RANGE))
    pos_size = int(np.mean([len(df) for df in train_user_dfs]))
    
    if session0:
        df = df_s0
        df = df.loc[df[df.columns[-1]].isin(S0_TRAIN_USER_RANGE)]
    else:
        df = df_s1
        df = df.loc[df[df.columns[-1]].isin(NEG_USER_RANGE)]

    # Sample negatives
    df = df.sample(n=pos_size * NEG_RATE, random_state=RANDOM_STATE)
    df[df.columns[-1]] = np.zeros((pos_size * NEG_RATE,))
    train_neg_df = df.copy()
    
    # Halve all user dataframes; use the first halves for training,
    # second halves for testing
    new_train = []
    test_user_dfs = []
    for df in train_user_dfs:
        df1, df2 = np.array_split(df, 2)
        new_train.append(df1)
        test_user_dfs.append(df2)
    train_user_dfs = new_train

    df1, _ = np.array_split(train_neg_df, 2)
    train_neg_df = df1

    if same_day:
        # We need to sample negatives in the same day scenario from session 0
        # in order to have consecutive steps
        df = df_s0
        test_neg_df = df.loc[df[df.columns[-1]].isin(S0_TEST_USER_RANGE)].copy()
    else:
        # Take the first half of every user dataframe from session 2
        test_user_dfs = list(map(lambda c: filter_df(df_s2, c), POS_USER_RANGE))
        test_user_dfs = list(map(lambda df: df.head(int(len(df)/2)), test_user_dfs))

        if session0:
            df = df_s0
            df = df.loc[df[df.columns[-1]].isin(S0_TEST_USER_RANGE)]
        else:
            df = df_s2
            df = df.loc[df[df.columns[-1]].isin(NEG_USER_RANGE)]

        test_neg_df = df.copy()

    return train_user_dfs, train_neg_df, test_user_dfs, test_neg_df
