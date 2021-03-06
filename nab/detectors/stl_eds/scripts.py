import pandas as pd
import os
from sys import path
import numpy as np
import argparse

path.append(os.path.abspath('/mnt/c/Users/maoltea/Documents/Time Series Anomaly Detection/stl'))
# path.append(os.path.abspath('/home/matei/Documents/timeseries/stl'))
path.append(os.path.abspath('/mnt/c/Users/maoltea/Documents/NAB'))

from stlLib import stl, infer_period, h_esd, threshold
import run

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


def frange(x, y, jump):
    """
    Range for floats
    """
    while x < y:
        yield x
        x += jump


if __name__ == '__main__':
    windows_path = 'labels/combined_windows.json'
    data_path = 'data'
    res_path = 'results/stlheds'
    null_path = 'results/null'

    parser = argparse.ArgumentParser()

    parser.add_argument("--detect",
                    help="Generate detector results but do not analyze results "
                    "files.",
                    default=False,
                    action="store_true")

    parser.add_argument("--optimize",
                    help="Optimize the thresholds for each detector and user "
                    "profile combination",
                    default=False,
                    action="store_true")

    parser.add_argument("--score",
                    help="Analyze results in the results directory",
                    default=True,
                    action="store_true")

    parser.add_argument("--normalize",
                    help="Normalize the final scores",
                    default=True,
                    action="store_true")

    parser.add_argument("--skipConfirmation",
                    help="If specified will skip the user confirmation step",
                    default=True,
                    action="store_true")

    parser.add_argument("--dataDir",
                    default="data",
                    help="This holds all the label windows for the corpus.")

    parser.add_argument("--resultsDir",
                    default="results",
                    help="This will hold the results after running detectors "
                    "on the data")

    parser.add_argument("--windowsFile",
                    default=os.path.join("labels", "combined_windows.json"),
                    help="JSON file containing ground truth labels for the "
                         "corpus.")

    parser.add_argument("-d", "--detectors",
                    nargs="*",
                    type=str,
                    default=["null", "numenta", "random", "skyline",
                             "bayesChangePt", "windowedGaussian", "expose",
                             "relativeEntropy", "earthgeckoSkyline"],
                    help="Comma separated list of detector(s) to use, e.g. "
                         "null,numenta")

    parser.add_argument("-p", "--profilesFile",
                    default=os.path.join("config", "profiles.json"),
                    help="The configuration file to use while running the "
                    "benchmark.")

    parser.add_argument("-t", "--thresholdsFile",
                    default=os.path.join("config", "thresholds.json"),
                    help="The configuration file that stores thresholds for "
                    "each combination of detector and username")

    parser.add_argument("-n", "--numCPUs",
                    default=None,
                    help="The number of CPUs to use to run the "
                    "benchmark. If not specified all CPUs will be used.")

    args = parser.parse_args()

    for tup in get_files(data_path):
        subdir, file = tup
        d = subdir.split('/')[-1]
        data = pd.read_csv(os.path.join(subdir, file))
        values = data['value']
        period = infer_period(values)
        if period < 2 or period >= values.shape[0]/2:
            season = np.zeros_like(values)
            period = 1
        else:
            _, season, _, _ = stl(y=values, p=period, robust=False)
        rest = values - season - np.median(values)
        anomalies_idx = h_esd(rest, k=0.002)
        data['anomaly_score'] = np.zeros_like(data['value'])
        data.loc[anomalies_idx, ['anomaly_score']] = 1.0
        d = subdir.split('/')[-1]
        labels = pd.read_csv(os.path.join(os.path.join(null_path, d), 'null_' + file))['label']
        data['label'] = labels
        data.to_csv(os.path.join(os.path.join(res_path, d), 'stlheds_' + file), index=False)
    run.main(args)
