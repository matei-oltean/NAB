import pandas as pd
import os
from sys import path
import numpy as np


path.append(os.path.abspath('/mnt/c/Users/maoltea/Documents/Time Series Anomaly Detection/stl'))
#  path.append(os.path.abspath('/home/matei/Documents/timeseries/stl'))


from stlLib import stl, inferPeriod, h_esd


def get_files(path):
    result = []
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                result.append([subdir, file])
    return result


"""
def load_json(path):
    json_data = open(path)
    return json.load(json_data)
"""


if __name__ == '__main__':
    windows_path = '../../../labels/combined_windows.json'
    data_path = '../../../data'
    res_path = '../../../results/stlheds'
    null_path = '../../../results/null'
    for tup in get_files(data_path):
        subdir, file = tup
        d = subdir.split('/')[-1]
        print(file)
        data = pd.read_csv(os.path.join(subdir, file))
        values = data['value']
        period = inferPeriod(values)
        if period == 0:
            anomalies_idx = []
        else:
            r, season, trend, w = stl(y=values, np=period)
            rest = values - season - trend# np.median(values)
            anomalies_idx = h_esd(rest, k=0.002)
        data['anomaly_score'] = np.zeros_like(data['value'])
        data.loc[anomalies_idx, ['anomaly_score']] = 1.0
        d = subdir.split('/')[-1]
        labels = pd.read_csv(os.path.join(os.path.join(null_path, d), 'null_' + file))['label']
        data['label'] = labels
        if labels.shape[0] != data.shape[0]:
            print('toto')
        data.to_csv(os.path.join(os.path.join(res_path, d), 'stlheds_' + file), index=False)