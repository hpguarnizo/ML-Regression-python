import pandas as pd 
import numpy as np 

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

#Función de validación
def rmsle_cv(model, n_folds, train_ds, target):
    """Para las aproximaciones de regresión, cross_val_score añadiendo una línea de codigo que mezcle los datos """
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train_ds.values) #Línea de codigo que mezcla los datos
    rmse= np.sqrt(-cross_val_score(model, 
                                   train_ds.values, 
                                   target, 
                                   scoring="neg_mean_squared_error", 
                                   cv = kf))
    return(rmse)


#Modelos base más robustos a valores atipicos, usando robust scaler: Lasso y ElasticNet. TODO
def robust_model(models):
    pass
        



class Models:
    def __init__(self):
        self.reg = {
            'ELASTIC_NET': ElasticNet(l1_ratio=.9, random_state=3),
            'GRADIENT': GradientBoostingRegressor(n_estimators=3000,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5),
            'LASSO': Lasso(random_state=1),
            'KERNEL_RIDGE': KernelRidge(kernel='polynomial', degree=2, coef0=2.5),
            'XGB': xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1),
            'LGB': lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              n_estimators=720, max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
        }

        self.params = {
            'ELASTIC_NET': {
                'alpha': [0.0005, 0.005, 1]
            }, 
            'GRADIENT': {
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'LASSO': {
                'alpha': [0.0005, 0.005, 1]
            },
            'KERNEL_RIDGE': {
                'alpha': [0.1, 0.5, 0.6]
            },
            'XGB': {
                'learning_rate': [0.05, 0.06, 0.07]
            },
            'LGB': {
                'learning_rate': [0.05, 0.06, 0.07]
            }
        }


    def grid_training(self, X, y, name):
        best_model = None

        reg_dic = self.reg[name]

        grid_reg = GridSearchCV(reg_dic, self.params[name], cv=3)
        grid_reg.fit(X, y.values.ravel())
        score = np.abs(grid_reg.best_score_)

        best_model = grid_reg.best_estimator_

        return best_model


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)