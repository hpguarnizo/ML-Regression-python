from utils import Utils
from model import Models
import pandas as pd

if __name__=='__main__':
    utils = Utils()
    models = Models()
    
    train = utils.load_data('./csv/clean_train.csv')
    test = utils.load_data('./csv/clean_test.csv')
    test.drop('SalePrice', axis=1, inplace=True)
    #X, y = utils.features_target(train, ['SalePrice'], ['SalePrice'])
    #best_model, best_score = models.grid_training(X,y)
    #print(best_model)
    #print(best_score)