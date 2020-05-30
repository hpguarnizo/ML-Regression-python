from data import Data
from utils import norm_target, scatter_plot, qq_plot, data_corr

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

    #scatter_plot(train, 'GrLivArea', 'SalePrice')

    train['SalePrice'] = data.train_log_transform('SalePrice')

    #norm_target(train, 'SalePrice')
    #qq_plot(train, 'SalePrice')

    #feature enginnering

    all_data, missing_data = data.all_data_missing(train, test, 'SalePrice')

    #data_corr(train)

    all_data = data.fill_na(['PoolQC', 
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
                            'MSSubClass' ],
                            all_data)

    all_data = data.fill_zero(['GarageType', 
                               'GarageFinish', 
                               'GarageQual', 
                               'GarageCond',
                               'BsmtFinSF1', 
                               'BsmtFinSF2', 
                               'BsmtUnfSF',
                               'TotalBsmtSF', 
                               'BsmtFullBath', 
                               'BsmtHalfBath',
                               'MasVnrArea'],
                               all_data)

    all_data = data.fill_most_frequent(['MSZoning',
                                        'Electrical',
                                        'KitchenQual',
                                        'Exterior1st',
                                        'Exterior2nd',
                                        'SaleType'],
                                        all_data)

    all_data = data.group_by(all_data, 'LotFrontage', 'Neighborhood')

    all_data = data.data_replace('Functional', 'Typ', all_data)

    all_data = data.drop_feature(['Utilities'], all_data)

    all_data = data.transform_num_cat(['MSSubClass', 
                                       'OverallCond',
                                       'YrSold',
                                       'MoSold' ],
                                       all_data)

    all_data = data.label_encoding(['FireplaceQu', 
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
                                    'MoSold'],
                                    all_data)
    

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    all_data = data.skew_features(all_data)

    all_data = data.dummy_features(all_data)

    train = all_data[:ntrain]
    test = all_data[ntrain:]

    data.to_csv(train)
    data.to_csv(test, split='test')

