from utils import Utils
from model import Models
import pandas as pd

if __name__=='__main__':
    utils = Utils()
    models = Models()
    
    train = utils.load_data('./csv/clean_train.csv')
    test = utils.load_data('./csv/clean_test.csv')

    X, y = utils.features_target(train, ['SalePrice'], ['SalePrice'])
    best_model = models.grid_training(X,y,'ELASTIC_NET')
    print(best_model)