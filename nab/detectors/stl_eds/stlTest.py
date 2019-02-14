from os.path import abspath
from sys import path
import pandas as pd
import numpy as np

path.append(abspath('/mnt/c/Users/maoltea/Documents/Time Series Anomaly Detection/stl'))

from stlLib import stl

if __name__ == '__main__':
    data = pd.read_csv('co2', delim_whitespace=True, header=None, dtype=np.float64).values.flatten()

    n = 348
    np = 12
    ns = 35
    nt = 19
    nl = 13
    no = 2
    ni = 1
    nsjump = 4
    ntjump = 2
    nljump = 2
    isdeg = 1
    itdeg = 1
    ildeg = 1

    rw, season, trend, work = stl(data, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, no)
    print(rw)
    print(help(stl))
