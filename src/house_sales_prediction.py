'''
#############################################################################################################
                                                                                                            #
    FileName      [house_sales_prediction.py]                                                               #
                                                                                                            #
    Synopsis      [This is a solution of a Kaggle competition: House Prices: Advanced Regression Tecniques. #
                   Our goal is to predict the sales prices from test set. This program engineers the train  #
                   and test set, create models and predicts the prices for the house in test set.]          #
                                                                                                            #
    Kaggle Score  [0.11297 - Top 3% on leaderboard]                                                         #
                                                                                                            #
    Author        [Leonardo Picchiami]                                                                      #
                                                                                                            #
#############################################################################################################
'''


import warnings

def warning_ignore(*args, **kwargs):
    pass
warnings.warn = warning_ignore

import pandas as pd
import numpy as np

from scipy.stats import skew, norm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor


class DataFrameRegressionExploration(object):
    def __init__(self, train, test):
        self.__train = train
        self.__test = test
        self.__train_test = None

    def set_train_test(self, train_test):
        self.__train_test = train_test

    def train_head(self):
        print()
        print(self.__train.head())
        print()

    def sale_price_distplot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))
        sns.distplot(self.__train.SalePrice, fit = norm, ax = ax1)        
        sns.distplot(np.log1p(self.__train.SalePrice), fit = norm, ax = ax2)
        plt.show()

    def outliers_plot_study(self):
        fig, ax = plt.subplots()
        ax.scatter(x = self.__train['GrLivArea'], y = self.__train['SalePrice'])
        plt.ylabel('SalePrice', fontsize = 13)
        plt.xlabel('GrLivArea', fontsize = 13)
        plt.show() 

    def sale_price_features_correlation(self):
        print("Correlation value of each feature with SalePrice:")
        correlation = self.__train.corr()["SalePrice"]
        correlation = correlation[np.argsort(correlation, axis = 0)[::-1]]
        print(correlation.to_string() + "\n")
    
    def correlation_to_multicollinearity_plot(self):
        corr = self.__train.corr()
        ten_max_corr = corr.nlargest(10, 'SalePrice')['SalePrice'].index
        coeff = np.corrcoef(self.__train[ten_max_corr].values.T)
        sns.set(font_scale = 1.25)
        hm = sns.heatmap(coeff, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = ten_max_corr.values, xticklabels = ten_max_corr.values)
        plt.show()

    def correlation_and_multicollinearity(self):
        print("\nDrop the features that have a low correlation with SalePrice and high correlation (multicollinearity) whth other features...")
        print()
        self.sale_price_features_correlation()
        self.correlation_to_multicollinearity_plot()
        print("Features that respect this condition are: 1stFlrSF and GarageArea\n")

    

class DataFrameRegression(object):
    def __init__(self, train, test):
        self.__train_path = train
        self.__test_path = test
        self.__train = None
        self.__test = None
        self.__train_test_set = None
        self.__target = None
        self.__training = None
        self.__testing = None
        self.__data_exploration = None

    def read_csv(self):
        print("Read train and test csv files...")
        self.__train = pd.read_csv(self.__train_path)
        self.__test = pd.read_csv(self.__test_path)
        self.__data_exploration = DataFrameRegressionExploration(self.__train, self.__test)
        self.__data_exploration.train_head()

    def remove_outliers(self):
        print("Remove outiliers from train set...")
        self.__data_exploration.outliers_plot_study()
        self.__train = self.__train[~((self.__train['GrLivArea'] > 4000) & (self.__train['SalePrice'] < 300000))]
        self.__data_exploration.outliers_plot_study()

    def concat_train_and_test_set(self):
        self.__train_test_set = pd.concat((self.__train.loc[:, 'MSSubClass' : 'SaleCondition'], self.__test.loc[:, 'MSSubClass' : 'SaleCondition']))
        print("Distribution SalePrice study...")        
        self.__data_exploration.sale_price_distplot()
        self.__train['SalePrice'] = np.log1p(self.__train['SalePrice'])
        self.__target = self.__train.SalePrice

    def drop_multicollinearity_and_needless_predictors(self):
        self.__data_exploration.correlation_and_multicollinearity()
        self.__train_test_set.drop(['1stFlrSF', 'GarageArea'], axis = 1, inplace = True)

        #All Utilities features records are "AllPub", except for one "NoSeWa" and two NA. Since record with 'NoSewa' is only in the training set, 
        #it can be removed from features to predictive model.
        self.__train_test_set.drop('Utilities', axis = 1, inplace = True)

    def modification_skewed_features(self):
        print("Skewed features normalizzation...")
        numeric_features = self.__train_test_set.dtypes[self.__train_test_set.dtypes != 'object'].index
        skewed_features = self.__train[numeric_features].apply(lambda x : skew(x.dropna()))
        skewed_features = skewed_features[skewed_features > 0.65]
        skewed_features = skewed_features.index
        self.__train_test_set[skewed_features] = np.log1p(self.__train_test_set[skewed_features])

    def get_dummies(self):
        print("Get the dummies variables...")
        self.__train_test_set = pd.get_dummies(self.__train_test_set)

    def drop_missing_value(self):
        print("Drop NA value from all dataset...")
        self.__train_test_set = self.__train_test_set.fillna(self.__train_test_set.mean())

    def divide_all_data(self):
        self.__training = self.__train_test_set[:self.__train.shape[0]]
        self.__testing = self.__train_test_set[self.__train.shape[0]:]

    def get_training(self):
        return self.__training

    def get_testing(self):
        return self.__testing

    def get_target(self):
        return self.__target

    def write_csv(self, result_prediction):
        print("Write the result prediction on csv file...")
        result = pd.DataFrame({'Id' : self.__test.Id, 'SalePrice' : result_prediction})
        result.to_csv('../result_dataset/result.csv', index = False)
    


class ModelPrediction(object):
    def LassoModel(self):
        return Lasso(alpha = 0.00035)

    def LassoPipelineModel(self):
        return make_pipeline(RobustScaler(), Lasso(alpha = 0.00035, random_state = 1))

    def GBoostModel(self):
        return GradientBoostingRegressor(loss = 'huber', learning_rate = 0.05, n_estimators = 8000, min_samples_split = 10, min_samples_leaf = 15, max_depth = 4, random_state = 6, max_features = 'sqrt')

    def GBoostPipelineModel(self):
        return make_pipeline(StandardScaler(), GradientBoostingRegressor(loss = 'huber', learning_rate = 0.05, n_estimators = 8000, min_samples_split = 10, min_samples_leaf = 15, max_depth = 4, random_state = 6, max_features = 'sqrt')) 



class AveragingPredictions(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.__models = models
        
    def fit(self, train, target):
        self.__models_cloned = [] 
        for mod in self.__models:
            self.__models_cloned.append(clone(mod))

        for model in self.__models_cloned:
            model.fit(train, target)
    
    def predict(self, train):
        model_predictions = []
        for model in self.__models_cloned:
            model_predictions.append(model.predict(train))
        predictions = np.column_stack(model_predictions)
        return np.mean(predictions, axis=1) 



def main():
    #Initializing object for regression prediction
    dataframe = DataFrameRegression("../dataset/train.csv", "../dataset/test.csv")
    models = ModelPrediction()
    
    #Data tidyng and features engineering
    dataframe.read_csv()
    dataframe.remove_outliers()
    dataframe.concat_train_and_test_set()
    dataframe.drop_multicollinearity_and_needless_predictors()
    dataframe.modification_skewed_features()
    dataframe.get_dummies()
    dataframe.drop_missing_value()
    dataframe.divide_all_data()

    #Machine learning modeling
    print("Creation of each model...")
    lasso = models.LassoModel()
    lasso_pipeline = models.LassoPipelineModel()
    GBoost = models.GBoostModel()
    GBoost_pipeline = models.GBoostPipelineModel()
    
    #Models fit and predictions
    print("Fit each model, predict on test set and calculate average of predictions...")
    averaging = AveragingPredictions(models = (GBoost, GBoost_pipeline, lasso, lasso_pipeline))
    averaging.fit(dataframe.get_training(), dataframe.get_target())
    prediction = np.expm1(averaging.predict(dataframe.get_testing()))
    
    #Writing the result of predictions
    dataframe.write_csv(prediction)
    
    print('Done.')


#Program main
if __name__ == '__main__': 
    main()
