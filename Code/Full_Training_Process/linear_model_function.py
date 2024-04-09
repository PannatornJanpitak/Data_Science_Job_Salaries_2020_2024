
"""

This file contain all function for creating linear model for [Data Science Job Salaries 2020 - 2024] dataset

"""
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from joblib import dump

#model parameter
def define_model_params():
    """
    #This function contain linear model parameter to do gridsearch in fine-tuning process
    #1.Linear Regression model
    #2.Ridge Regression model
    #3.Lasso Regression model
    #4.ElasticNets Regression model
    #5.Random Forest Regression model
    #6.Support Vector Machine Regression model
    input = []
    output = model parameters in this function
    """
    model_params = {
    'linear': {
        'model': LinearRegression(),
        'params': {
            'fit_intercept': [True, False]
            }
        },
    'ridge': {
        'model': Ridge(),  
        'params': {
            'alpha': [10],
            'fit_intercept': [True]
            }
        },
    'lasso': {
        'model': Lasso(),
        'params': {
            'alpha': [20],
            'fit_intercept': [True],
            'selection': ['random', 'cyclic']
            }
        },
    'elasticnet': {
        'model': ElasticNet(), 
        'params': {
            'alpha': [20],
            'l1_ratio': [1.0], 
            'selection': ['random', 'cyclic'],
            'fit_intercept': [True]
            }
        },
    'random_forest': {
        'model': RandomForestRegressor(), 
        'params': {
            'n_estimators': [100] 
            }
        },
    'SVM': {
        'model': SVR(), 
        'params': {
            'C': [1],
            'kernel': ['linear'],
            'gamma': ['scale', 'auto'],
            }
        }
    }
    return model_params


#train machine learning model with Gridsearch
def find_best_linear_model(X, y):
    """
    #This function perform gridsearch in fine-tuning process by using parameters from [define_model_params()] function
    input = (X,y) = (your data feature, your data target)
    output = dataframe result of this model, best parameter of each model
    """
    #Get model parameters
    model_params = define_model_params()
    results = []
    linear_models = {}

    #Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    for model_name, model_param in model_params.items():
        #GridSearch
        gs = GridSearchCV(model_param['model'], model_param['params'], cv=5, return_train_score=False)
        gs.fit(X_scaled, y)
        #record result
        linear_models[model_name] = gs.best_estimator_
        results.append({
            "model": model_name,
            "best_score": gs.best_score_,
            "best_params": gs.best_params_
        })
    #convert gridsearch result to dataframe    
    df_result = pd.DataFrame(results, columns=['model', 'best_score', 'best_params'])

    #save best linear models
    save_best_linear_model(df_result, linear_models)

    return  df_result, linear_models

#Save best linear model
def save_best_linear_model(df_result, linear_models):
    best_linear_model =  linear_models[df_result.model[df_result.best_score.idxmax()]]
    dump(best_linear_model, 'linear_best_model.pkl')

