import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import scipy.stats as ss 
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder

from utils import plot_miss_val
from utils import norm_target, scatter_plot, qq_plot

class Data:
    def __init__(self, train_path, test_path):
        self.train = pd.read_csv(train_path)
        self.test =  pd.read_csv(test_path)

    def train_disp(self):
        return print(self.train.head())

    def test_disp(self):
        return print(self.test.head())

    def drop_train_id(self, show=False):
        train_ID = self.train['Id']
        self.train.drop("Id", axis = 1, inplace = True)
        
        if show:
            print("\nThe train data size before dropping Id feature is : {} ".format(self.train.shape))
            print("\nThe train data size after dropping Id feature is : {} ".format(self.train.shape))
        
        return train_ID

    def drop_test_id(self, show=False):
        test_ID = self.test['Id']
        self.test.drop("Id", axis = 1, inplace = True)

        if show:
            print("\nThe test data size before dropping Id feature is : {} ".format(self.test.shape))
            print("\nThe test data size after dropping Id feature is : {} ".format(self.test.shape))
        
        return test_ID

    def train_del_outliers(self, column1_set, column2, x_lim_set, y_lim, plot=False):
        for column1 in column1_set:
            for x_lim in x_lim_set:
                self.train.drop(self.train[(self.train[column1]>x_lim) & (self.train[column2]<y_lim)].index, inplace=True)

        if plot:
            scatter_plot(train, [column1], [column2])

        return self.train

    def train_log_transform(self, target, plot=False):
        ds = np.log1p(self.train[target])

        if plot:
            norm_target(train, target)
            qq_plot(train, target)
        
        return ds

    def target(self, df, target):
        return df[target].values

    def all_data_missing(self, ds, plot=False, show=False):
        all_data_na = (ds.isnull().sum() / len(ds)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        
        if plot:
            plot_miss_val(all_data_na)
        
        if show:
            print(missing_data.head(20))

        return ds, missing_data

    def fill_na(self, missing_data, df):
        for col in missing_data:
            df[col] = df[col].fillna("None")
        
        return df

    def group_by(self, df, column1, column2):
        df[column1] = df.groupby(column2)[column1].transform(lambda x: x.fillna(x.median()))
        
        return df

    def fill_zero(self, zero_data, df):
        for col in zero_data:
            df[col] = df[col].fillna(0)
        
        return df

    def drop_feature(self, features, df):
        return df.drop(features, axis=1)

    def data_replace(self, feature, replace, df):
        df[feature] = df[feature].fillna(replace)
        
        return df

    def fill_most_frequent(self, data, df):
        for col in data:
            df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def transform_num_cat(self, data, df):
        for col in data:
            df[col] = df[col].astype(str)
        
        return df 

    def label_encoding(self, data, df):
        for col in data:
            lbl = LabelEncoder() 
            lbl.fit(list(df[col].values)) 
            df[col] = lbl.transform(list(df[col].values))

        return df

    def skew_features(self, df, verbose=False):
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

        skewness = pd.DataFrame({'Skew' :skewed_feats})   

        if verbose:
            print("\nSkew in numerical features: \n")
            print(skewness.head(10))

        skewness = skewness[abs(skewness) > 0.75]

        if verbose:
            print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        skewed_features = skewness.index
        lam = 0.15   

        for feat in skewed_features:
            #df[feat] += 1
            df[feat] = boxcox1p(df[feat], lam)

        #df[skewed_features] = np.log1p(all_data[skewed_features])

        return df

    def dummy_features(self, df):
        return pd.get_dummies(df)

    def to_csv(self, df_train, df_test, index, split='train'):
        if split == 'train':
            drop_col = [list(df_train.columns)[i] for i in range(len(list(df_train.columns))-1) if list(df_train.columns)[i] not in df_test.columns and list(df_train.columns)[i] != 'SalePrice'] 
            df = df_train.drop(drop_col, axis=1)
            #df = df_train
            df = pd.concat([index, df], axis=1)
            self.check_missing_data(df)
            #print(df.head())
        else:
            df = df_test
            df = pd.concat([index, df], axis=1)
            #print(df.head())  
        return df.to_csv('../csv/clean_'+split+'.csv', index=False)

    def check_missing_data(self, df):
        df_na = (df.isnull().sum() / len(df)) * 100
        df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Missing Ratio' :df_na})
        
        return print(missing_data)