from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from util.const import *

def get_csv(session, split):
    return 'zju_gaitaccel_session_' + str(session) + '_' + str(split) + '.csv'

def get_X_y(df_s, pos_size):
    pos = (pos_size == 0)
    user_range = POS_USER_RANGE if pos else NEG_USER_RANGE
    allowed = ['u%03d' % i for i in user_range]

    df = df_s.loc[df_s[df_s.columns[-1]].isin(allowed)]
    if pos:
        y = df[df.columns[-1]].values
        y = LabelEncoder().fit_transform(y) + 1
    else:
        df = df.sample(n=pos_size * NEG_RATE, random_state=RANDOM_STATE)
        y = np.zeros((pos_size * NEG_RATE,))
    X = df.drop([df.columns[-1]], axis=1).values

    return X, y

def transform_labels(df):
    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    df[df.columns[-1]] = y
    
    return df

def get_df(df, pos_class):
    return df.loc[df[df.columns[-1]] == pos_class]

def load(raw, same_day):
    df_s1 = pd.read_csv(Path(FEAT_PATH).joinpath(Path(get_csv(1, 128))), header=None)
    df_s2 = pd.read_csv(Path(FEAT_PATH).joinpath(Path(get_csv(2, 128))), header=None)

    df_s1 = transform_labels(df_s1)
    df_s2 = transform_labels(df_s2)

    train_user_dfs = list(map(lambda c: get_df(df_s1, c), POS_USER_RANGE))
    pos_size = sum([len(df) for df in train_user_dfs])
    df = df_s1
    df = df.loc[df[df.columns[-1]].isin(NEG_USER_RANGE)]
    df = df.sample(n=pos_size * NEG_RATE, random_state=RANDOM_STATE)
    df[df.columns[-1]] = np.zeros((pos_size * NEG_RATE,))
    train_neg_df = df.copy()

    new_train = []
    test_user_dfs = []
    for df in train_user_dfs:
        df1, df2 = np.array_split(df, 2)
        new_train.append(df1)
        test_user_dfs.append(df2)

    df1, df2 = np.array_split(train_neg_df, 2)
    train_neg_df = df1
    test_neg_df = df2
    
    if not same_day:
        test_user_dfs = list(map(lambda c: get_df(df_s2, c), POS_USER_RANGE))
        pos_size = sum([len(df) for df in test_user_dfs])
        df = df_s2
        df = df.loc[df[df.columns[-1]].isin(NEG_USER_RANGE)]
        df = df.sample(n=pos_size * NEG_RATE, random_state=RANDOM_STATE)
        df[df.columns[-1]] = np.zeros((pos_size * NEG_RATE,))
        test_neg_df = df.copy()

    return train_user_dfs, train_neg_df, test_user_dfs, test_neg_df
