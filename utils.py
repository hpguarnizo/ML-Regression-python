import pandas as pd 


class Utils():
    def load_data(self, path):
        return pd.read_csv(path)

    def features_target(self, ds, dropcols, y):
        X = ds.drop(dropcols, axis=1)
        y = ds[y]
        return X, y
