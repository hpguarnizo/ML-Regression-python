from utils import Utils
from model import Models, AveragingModels
import pandas as pd

if __name__=='__main__':
    utils = Utils()
    models_ = Models()
    
    X = utils.load_data('./csv/clean_train.csv')
    test = utils.load_data('./csv/clean_test.csv')
    y = utils.load_data('./csv/target.csv')

    test_id = test['Id']
    test.drop('Id', axis=1, inplace=True)

    models = ['ELASTIC_NET', 'GRADIENT', 'LASSO', 'KERNEL_RIDGE', 'XGB']
    base_models = []

    for model in models:
        base_model = models_.grid_training(X,y,model)
        base_models.append(base_model)
    
    averaged_models = AveragingModels(models = (base_models[0], 
                                                base_models[1], 
                                                base_models[2], 
                                                base_models[3]))

    n_folds = 5

    score = utils.rmsle_cv(averaged_models, n_folds, X, y)
    print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

    averaged_models.fit(X, y)
    pred = averaged_models.predict(test)

    sub = utils.make_sub(pred, test_id)
    print(sub.describe())

    sub.to_csv('./csv/submission.csv', index=False)