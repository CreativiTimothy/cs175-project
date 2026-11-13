import numpy as np
import pandas as pd


def get_data(file_name):
    df = pd.read_csv(file_name)
    df.columns = df.columns.str.strip().str.lower()

    # can change this based on what the input data file calls it
    cols = ['date', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in cols):
        raise ValueError('csv needs date, open, high...')
    
    # for now i just want to do one stock. change this later
    if 'symbol' in df.columns and df['symbol'].nunique() > 1:
        raise ValueError('just using one stock for now')
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date')

    # some things to prevent training on bad data. will probably need to add more to this but this is all that i can think of for now
    if df[['open', 'high', 'low', 'close']].isna().any().any():
        raise ValueError('prices are missing')
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('should be positive')
    
    return df


def make_data(df):
    prices = df [['open', 'high', 'low', 'close']].to_numpy()
    X = []
    y = []

    for i in range(100, len(prices)):
        past = prices[i - 100: i]
        last_close = past[-1, 3]

        past = past / last_close - 1
        X.append(past.flatten())
        y.append(int(prices[i, 3] > last_close))

    return np.array(X), np.array(y)


def split_data(X, y):
    n = len(y)
    a = int(n * 0.7)
    b = int (n * 0.85)

    X_tr, X_val, X_te = X[: a], X[a: b], X[b: ]
    y_tr, y_val, y_te = y[: a], y[a: b], y[b: ]

    # i doubt this would be an issue but just in case i want to make sure that our training data has examples of both up and down stock movement
    if len(np.unique(y_tr)) < 2:
        raise ValueError('training data needs both up and down days')
    
    return X_tr, X_val, X_te, y_tr, y_val, y_te


if __name__ == '__main__':
    raise ValueError('put data file name below and comment this out')
    df = get_data('data file here')
    X, y = make_data(df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y)

    print('train: ', X_tr.shape)
    print('val: ', X_val.shape)
    print('test: ', X_te.shape)