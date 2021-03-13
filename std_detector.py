from typing import List

import pandas as pd


class StdDetector:
    """
    Usage:
    model = StdDetector(numeric_cols, 3)
    model.train(train)
    anomalies = model.detect(train)  # Dataframe where True means anomaly
    """

    def __init__(self, cols: List[str], tolerance=2):
        self.cols = cols
        self.tolerance = tolerance
        self._means = None
        self._stds = None

    def train(self, df):
        self._means = df[self.cols].mean()
        self._stds = df[self.cols].std()

    def detect(self, df):
        if self._means is None: raise RuntimeError('Must train first!')

        lower = self._means - self.tolerance * self._stds
        upper = self._means + self.tolerance * self._stds

        anomalies = {}
        for col_name in self.cols:
            anomalies[col_name] = (df[col_name] <= lower[col_name]) | (df[col_name] >= upper[col_name])
        return pd.DataFrame(anomalies)
