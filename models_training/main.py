import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import logging
import numpy as np
from sklearn import metrics
from xgboost import XGBRegressor


def train_model_RandomForestRegressor(X, Y,filename):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(max_depth=16)
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_test)

    logger.info(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, preds_valid)}')
    logger.info(f'Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, preds_valid)}')
    logger.info(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(y_test, preds_valid))}')
    mape = metrics.mean_absolute_percentage_error(y_test,preds_valid)
    logger.info(f'Mean Absolute Percentage Error (MAPE):{mape}')
    logger.info(f'Accuracy: {1 - mape}')
    with open(f'../models_pickle/model_{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)

def train_model_XGBoost(X,Y,filename):
    params = {
        'n_estimators': 100,
        'max_depth': 25,
        'learning_rate': 0.1,
    }
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_test)

    # log first five predictions
    logger.info(f'Predictions: {preds_valid[:5]}')
    # log first five test values
    logger.info(f'Real: {y_test[:5]}')
    logger.info(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, preds_valid)}')
    logger.info(f'Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, preds_valid)}')
    logger.info(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(y_test, preds_valid))}')
    mape = metrics.mean_absolute_percentage_error(y_test, preds_valid)
    logger.info(f'Mean Absolute Percentage Error (MAPE):{mape}')
    logger.info(f'Accuracy: {1 - mape}')
    with open(f'../models_pickle/model_XGBoost_{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log')
    logger = logging.getLogger('train_model')



    # Load data
    logger.info('Loading data...')
    csv = pd.read_csv('../csv/Completo.csv', sep=';', encoding='latin-1')
    logger.info('Data loaded')
    logger.info('Preprocessing data...')
    csv.dropna(inplace=True)
    logger.info('Data preprocessed')

    colunas_remover = [x for x in csv.columns[0:51] if x not in ['TP_ESCOLA','TP_FAIXA_ETARIA','TP_ENSINO']]

    X = csv.drop(columns=colunas_remover)
    y_nota_ch = csv['NU_NOTA_CH'] #predict column
    y_nota_cn = csv['NU_NOTA_CN'] #predict column
    y_nota_lc = csv['NU_NOTA_LC'] #predict column
    y_nota_mt = csv['NU_NOTA_MT'] #predict column
    y_nota_redacao = csv['NU_NOTA_REDACAO'] #predict column


    mymap = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6,'G':7,'H':8, 'I':9, 'J':10, 'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P': 16, 'Q':17}

    logger.info('Ajustandando arquivo')
    X = X.map(lambda s: mymap.get(s) if s in mymap else s)
    logger.info('Arquivo ajustado')

    logger.info('Training models...')
    # Train models
    logger.info('Training model nota_ch...')
    start_time = datetime.now()
    train_model_XGBoost(X, y_nota_ch, 'nota_ch')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_ch trained')

    logger.info('Training model nota_cn...')
    start_time = datetime.now()
    train_model_XGBoost(X, y_nota_cn, 'nota_cn')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_cn trained')

    logger.info('Training model nota_lc...')
    start_time = datetime.now()
    train_model_XGBoost(X, y_nota_lc, 'nota_lc')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_lc trained')

    logger.info('Training model nota_mt...')
    start_time = datetime.now()
    train_model_XGBoost(X, y_nota_mt, 'nota_mt')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_mt trained')

    logger.info('Training model nota_redacao...')
    start_time = datetime.now()
    train_model_XGBoost(X, y_nota_redacao, 'nota_redacao')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_mt nota_redacao')


    logger.info('Trained models')


