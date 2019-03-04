import numpy as np


class Three_Sigma:
    def __init__(self, window, mode='median'):
        """
        Initalises a 3-sigma detector
        Parameters
        ----------
        window: int
            Window size of the detector.
        
        mode: string
            'median' for using a median and mad, 'mean' for using a mean and variance.
        """
        self.data = np.zeros(window)
        self.window = window
        self.index = 0
        self.full = False # True if at least window observations have been seen
        self.mode = mode
    
    def detect(self, data):
        """
        Adds an observation to the accumulator
        Parameters
        ----------
        data: float
            New observation at time t.

        Returns
        -------
            True if the new data is anomalous, False if not (or if still not enough observations have been seen).
        """
        self.data[self.index] = data
        if self.index == self.window - 1:
            self.full = True
        if self.full:
            if self.mode == 'mean':
                res = self.anomalyMean()
            else:
                res = self.anomalyMedian()
        else:
            res = False
        self.index = (self.index + 1) % self.window
        return res

    def anomalyMean(self):
        """
        Returns true if the last added data was anomalous, using a mean.
        """
        running_mean = np.mean(self.data)
        running_std = np.std(self.data)
        if self.data[self.index] - running_mean > 3*running_std:
            return True
        return False

    def anomalyMedian(self):
        """
        Returns true if the last added data was anomalous, using a median.
        """
        running_median = np.median(self.data)
        mad = np.median(np.abs(self.data - running_median))
        sigma = 1.4826*mad
        if self.data[self.index] - running_median > 3*sigma:
            return True
        return False
