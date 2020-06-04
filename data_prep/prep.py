from data import Data 
from utils import data_corr

if '__main__' == __name__:
    data = Data('../csv/train.csv', '../csv/test.csv')
    #data.train_disp()
    #data.test_disp()

    train = data.train
    test = data.test

    ntrain = train.shape[0]

    train_id = data.drop_train_id()
    test_id = data.drop_test_id()

    train = data.train_del_outliers(['GrLivArea'], 'SalePrice', [4000], 300000)

    train['SalePrice'] = data.train_log_transform('SalePrice')
    target = train['SalePrice']

    #feature enginnering

    train, missing_data_train = data.all_data_missing(train)
    test, missing_data_test = data.all_data_missing(test)

    #data_corr(train)

    missing_data_nan = ['PoolQC', 
                          'MiscFeature', 
                          'Alley', 
                          'Fence',
                          'FireplaceQu',
                          'GarageType', 
                          'GarageFinish', 
                          'GarageQual', 
                          'GarageCond',
                          'BsmtQual', 
                          'BsmtCond', 
                          'BsmtExposure', 
                          'BsmtFinType1', 
                          'BsmtFinType2',
                          'MasVnrType',
                          'MSSubClass' ]

    missing_data_zero = ['GarageType', 
                         'GarageFinish', 
                         'GarageQual', 
                         'GarageCond',
                         'BsmtFinSF1', 
                         'BsmtFinSF2', 
                         'BsmtUnfSF',
                         'TotalBsmtSF', 
                         'BsmtFullBath', 
                         'BsmtHalfBath',
                         'MasVnrArea']
    
    missing_data_non_present = ['MSZoning',
                                'Electrical',
                                'KitchenQual',
                                'Exterior1st',
                                'Exterior2nd',
                                'SaleType']

    data_num_cat = ['MSSubClass', 
                    'OverallCond',
                    'YrSold',
                    'MoSold' ]

    data_label_encod = ['FireplaceQu', 
                        'BsmtQual', 
                        'BsmtCond', 
                        'GarageQual', 
                        'GarageCond', 
                        'ExterQual', 
                        'ExterCond',
                        'HeatingQC', 
                        'PoolQC', 
                        'KitchenQual', 
                        'BsmtFinType1', 
                        'BsmtFinType2', 
                        'Functional', 
                        'Fence', 
                        'BsmtExposure', 
                        'GarageFinish', 
                        'LandSlope',
                        'LotShape', 
                        'PavedDrive', 
                        'Street', 
                        'Alley', 
                        'CentralAir', 
                        'MSSubClass', 
                        'OverallCond', 
                        'YrSold', 
                        'MoSold']

    data_drop_feature = ['GarageYrBlt', 'GarageArea', 'GarageCars']

    train = data.fill_na(missing_data_nan,
                           train)
    
    test = data.fill_na(missing_data_nan,
                           test)

    train = data.fill_zero(missing_data_zero,
                              train)

    test = data.fill_zero(missing_data_zero,
                              test)

    train = data.fill_most_frequent(missing_data_non_present,
                                       train)

    test = data.fill_most_frequent(missing_data_non_present,
                                       test)

    train = data.group_by(train, 'LotFrontage', 'Neighborhood')

    test = data.group_by(test, 'LotFrontage', 'Neighborhood')

    train = data.data_replace('Functional', 'Typ', train)

    test = data.data_replace('Functional', 'Typ', test)

    train = data.drop_feature(['Utilities'], train)

    test = data.drop_feature(['Utilities'], test)

    train = data.transform_num_cat(data_num_cat,
                                      train)

    test = data.transform_num_cat(data_num_cat,
                                      test)

    train = data.label_encoding(data_label_encod,
                                    train)
    
    test = data.label_encoding(data_label_encod,
                                    test)

    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

    test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
    
    train = data.skew_features(train)

    test = data.skew_features(test)

    train = data.dummy_features(train)

    test = data.dummy_features(test)

    train = data.drop_feature(data_drop_feature, train)

    test = data.drop_feature(data_drop_feature, test)

    data.check_missing_data(train)

    data.check_missing_data(test)

    data.to_csv(train, test, train_id)
    data.to_csv(train, test, test_id, split='test')

    

