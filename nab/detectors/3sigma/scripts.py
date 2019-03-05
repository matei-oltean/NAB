import pandas as pd
import os
from sys import path
import numpy as np
import argparse
from three_sigma import detect_anomalies
from itertools import product as cartesian_product

path.append(os.path.abspath('/mnt/c/Users/maoltea/Documents/NAB'))
# path.append(os.path.abspath('/home/matei/Documents/NAB'))

import run


def get_files(path):
    result = []
    for subdir, _, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                result.append([subdir, file])
    return result


if __name__ == '__main__':
    windows_path = 'labels/combined_windows.json'
    data_path = 'data'
    res_path = 'results/3sigma'
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

    space = cartesian_product(['median', 'mean'], ['std', 'mad'], range(20, 51, 10))

    print('mode,std,window,score')

    for mode, std, window in space:
        for tup in get_files(data_path):
            subdir, file = tup
            d = subdir.split('/')[-1]
            # print(file)
            data = pd.read_csv(os.path.join(subdir, file))
            values = data['value']
            data['anomaly_score'] = detect_anomalies(data=values, window=window, mode=mode, std=std)
            d = subdir.split('/')[-1]
            labels = pd.read_csv(os.path.join(os.path.join(null_path, d), 'null_' + file))['label']
            data['label'] = labels
            data.to_csv(os.path.join(os.path.join(res_path, d), '3sigma_' + file), index=False)
        score = run.main(args)
        result = str.format('{},{},{},{}', mode, std, window, score)
        print(result)
