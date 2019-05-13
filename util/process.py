from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from util.const import *

def load_feat(csv):
    df = pd.read_csv(Path(PATH).joinpath(Path(csv)), header=None)

    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    df[df.columns[-1]] = y
    
    return df

def load_raw(csv):
    df = pd.read_csv(Path(PATH).joinpath(Path(csv)), sep='\t')

    y = df[df.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    df[df.columns[-1]] = y
    
    return df

def get_feat_csv(session, split):
    return 'zju_gaitaccel_session_' + str(session) + '_' + str(split) + '.csv'

def get_raw_csv(session, split):
    return 'zju_raw_session_' + str(session) + '_' + str(split) + '.csv'

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

def get_df(df, pos_class):
    return df.loc[df[df.columns[-1]] == pos_class]

def load_auto(raw):
    if raw:
        df_s0 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(0, 128))), sep='\t')
        df_s1 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(1, 128))), sep='\t')
        # df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_raw_csv(2, 128))), sep='\t')
    else:
        df_s0 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(0, 128))), header=None)
        df_s1 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(1, 128))), header=None)
        # df_s2 = pd.read_csv(Path(PATH).joinpath(Path(get_feat_csv(2, 128))), header=None)

    return pd.concat([df_s0, df_s1])

def load(raw, split, same_day):
    session0 = (split is None)
    if not session0:
        POS_USER_RANGE = range(1, split + 1)
        NEG_USER_RANGE = range(split + 1, 154)
    else:
        POS_USER_RANGE = range(1, 154)

    if raw:
        df_s0 = load_raw(get_raw_csv(0, 128))
        df_s1 = load_raw(get_raw_csv(1, 128))
        df_s2 = load_raw(get_raw_csv(2, 128))
    else:
        df_s0 = load_feat(get_feat_csv(0, 128))
        df_s1 = load_feat(get_feat_csv(1, 128))
        df_s2 = load_feat(get_feat_csv(2, 128))

    train_user_dfs = list(map(lambda c: get_df(df_s1, c), POS_USER_RANGE))
    pos_size = int(np.mean([len(df) for df in train_user_dfs]))
    if session0:
        df = df_s0
        df = df.loc[df[df.columns[-1]].isin(S0_TRAIN_USER_RANGE)]
    else:
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

    train_user_dfs = new_train
    df1, df2 = np.array_split(train_neg_df, 2)
    train_neg_df = df1
    test_neg_df = df2

    if not same_day:
        test_user_dfs = list(map(lambda c: get_df(df_s2, c), POS_USER_RANGE))
        test_user_dfs = list(map(lambda df: df.sample(frac=0.5, random_state=RANDOM_STATE), test_user_dfs))
        pos_size = int(np.mean([len(df) for df in test_user_dfs]))

        if session0:
            df = df_s0
            df = df.loc[df[df.columns[-1]].isin(S0_TEST_USER_RANGE)]
        else:
            df = df_s2
            df = df.loc[df[df.columns[-1]].isin(NEG_USER_RANGE)]

        df = df.sample(n=pos_size * NEG_RATE, random_state=RANDOM_STATE)
        df[df.columns[-1]] = np.zeros((pos_size * NEG_RATE,))
        test_neg_df = df.copy()

    return train_user_dfs, train_neg_df, test_user_dfs, test_neg_df
