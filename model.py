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


class Models:
    def __init__(self):
        self.reg = {
            'ELASTIC_NET': ElasticNet(),
            'GRADIENT': GradientBoostingRegressor(),
            'LASSO': Lasso(),
            'KERNEL_RIDGE': KernelRidge(),
            'XGB': xgb.XGBRegressor(),
            'LGB': lgb.LGBMRegressor()
        }

        self.params = {
            'ELASTIC_NET': {
                'alpha': [0.0005, 0.005, 1],
                'l1_ratio': 0.9,
                'random_state': 1
            }, 
            'GRADIENT': {
                'loss': 'huber', #Because outliers
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': 4,
                'min_samples_leaf': 15,
                'min_sample_split': 10
            },
            'LASSO': {
                'alpha': [0.0005, 0.005, 1],
                'random_state': 1
            },
            'KERNEL_RIDGE': {
                'alpha': [0.1, 0.5, 0.6],
                'kernel': 'polynomial',
                'degree': 2,
                'coef0': 2.5
            },
            'XGB': {
                'colsample_bytree': 0.4603,
                'gamma': 0.0468,
                'learning_rate': [0.05, 0.06, 0.07],
                'max_depth': 3,
                'min_child_weight': 1.7817,
                'n_estimators': 2200,
                'reg_alpha': 0.4640,
                'reg_lambda': 0.8571,
                'subsample': 0.5213,
                'silent': 1,
                'random_state': 7,
                'nthread': -1
            },
            'LGB': {
                'objective': 'regression', #Objective regression
                'num_leaves': 5,
                'learning_rate': [0.05, 0.06, 0.07],
                'n_estimators': 720,
                'max_bin': 55,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'feature_fraction': 0.2319,
                'feature_fraction_seed': 9,
                'bagging_seed': 9,
                'min_data_in_leaf': 6,
                'min_sum_hessian_in_leaf': 11
            }
        }

    #TODO: Terminar función que filtre los parametros para los modelos.
    def grid_training(self, X, y):
        best_score = 999
        best_model = None

        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv=3)
            grid_reg.fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        return best_model, best_score