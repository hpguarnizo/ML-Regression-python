import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

import scipy.stats as ss 
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder

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

    def train_del_outliers(self, column1_set, column2, x_lim_set, y_lim):
        for column1 in column1_set:
            for x_lim in x_lim_set:
                self.train.drop(self.train[(self.train[column1]>x_lim) & (self.train[column2]<y_lim)].index, inplace=True)

        return self.train

    def train_log_transform(self, target):
        return np.log1p(self.train[target])

    def target(self, df, target):
        return df[target].values

    def all_data_missing(self, train, test, target, plot=False, show=False):
        all_data = pd.concat((train, test)).reset_index(drop=True)
        all_data.drop(['SalePrice'], axis=1, inplace=True)
        
        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
        

        if plot:
            fig, ax = plt.subplots(figsize=(15, 12))
            plt.xticks(rotation='90')
            sns.set(); np.random.seed(0)
            ax = sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features', fontsize=15)
            plt.ylabel('Percent of missing values', fontsize=15)
            plt.title('Percent missing data by feature', fontsize=15)
            plt.show()
        
        if show:
            print(missing_data.head(20))

        return all_data, missing_data

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

        #all_data[skewed_features] = np.log1p(all_data[skewed_features])

        return df

    def dummy_features(self, df):
        return pd.get_dummies(df)

    def to_csv(self, df, split='train'):
        return df.to_csv('../csv/clean_'+split+'.csv', index=False)