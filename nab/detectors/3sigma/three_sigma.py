import numpy as np
import pandas as pd


def mad(data):
    median = np.nanmedian(data)
    return 1.4826*np.nanmedian(np.abs(data - median))


def online_var(data, limit=100):
    M2 = 0
    mean = 0
    result = []
    for i, x in enumerate(data[:max(limit, len(data))]):
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        M2 += delta*delta2
        result.append(M2/(i+1))
    return result


def online_std(data, limit=100):
    var = online_var(data, limit)
    std = np.sqrt(var)
    result = np.full_like(data, std[-1])
    result[0: len(var)] = std
    return result


def detect_anomalies(data: pd.Series, window: int, mode='mean', std='std', limit=100):
    rolling_data = data.rolling(window)
    estimator = rolling_data.mean() if mode == 'mean' else rolling_data.median()
    noise = data - estimator
    std = np.std(noise[window:]) if std == 'std' else mad(noise) # online_std(noise, limit)
    return (np.abs(noise) > 3*std).astype(int)
