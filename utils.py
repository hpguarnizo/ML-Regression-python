import pandas as pd 
from sklearn.model_selection import KFold, cross_val_score
import numpy as np 


class Utils():
    def load_data(self, path):
        return pd.read_csv(path)

    def features_target(self, ds, dropcols, y):
        X = ds.drop(dropcols, axis=1)
        y = ds[y]
        return X, y

    #Función de validación
    def rmsle_cv(self, model, n_folds, train_ds, target):
        """Para las aproximaciones de regresión, cross_val_score añadiendo una línea de codigo que mezcle los datos """
        kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train_ds.values) #Línea de codigo que mezcla los datos
        rmse= np.sqrt(-cross_val_score(model, 
                                    train_ds.values, 
                                    target, 
                                    scoring="neg_mean_squared_error", 
                                    cv = kf))
        return(rmse)


    def make_sub(self, prediction, index):
        sub = pd.DataFrame()
        sub['Id'] = index
        sub['SalePrice'] = np.expm1(prediction)

        return sub