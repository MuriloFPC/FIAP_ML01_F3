import logging
import os
import pickle
from datetime import datetime
import io
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor  # noqa


def train_model_RandomForestRegressor(X, Y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42)
    model = RandomForestRegressor(max_depth=16)
    model.fit(X_train, y_train)
    preds_valid = model.predict(X_test)

    logger.info(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(y_test, preds_valid)}')
    logger.info(f'Mean Squared Error (MSE): {metrics.mean_squared_error(y_test, preds_valid)}')
    logger.info(f'Root Mean Squared Error (RMSE): {np.sqrt(metrics.mean_squared_error(y_test, preds_valid))}')
    mape = metrics.mean_absolute_percentage_error(y_test, preds_valid)
    logger.info(f'Mean Absolute Percentage Error (MAPE):{mape}')
    logger.info(f'Accuracy: {1 - mape}')
    with open(f'../models_pickle/model_{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)


def train_model_XGBoost(X, Y, filename):
    params = {
        'max_depth': 10,
        'learning_rate': 0.1,
        'tree_method': 'hist',
        'device': 'cuda'  # Using GPU
    }

    # Categorical columns were converted to numeric values
    for col in X.select_dtypes(include=['category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Data for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Train model
    model = xgb.train(params, dtrain, num_boost_round=50)

    # Predict
    preds_valid = model.predict(dtest)

    # model = XGBRegressor(**params)
    # model.fit(X_train, y_train)
    # preds_valid = model.predict(X_test)

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

    output_dir = '../models_pickle/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save trained model
    with open(f'../models_pickle/model_XGBoost_{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='app.log')

    logger = logging.getLogger('train_model')

    dtype_dict = {
        'Q001': 'category',
        'Q002': 'category',
        'Q003': 'category',
        'Q004': 'category',
        'Q005': 'category',
        'Q006': 'category',
        'Q007': 'category',
        'Q008': 'category',
        'Q009': 'category',
        'Q010': 'category',
        'Q011': 'category',
        'Q012': 'category',
        'Q013': 'category',
        'Q014': 'category',
        'Q015': 'category',
        'Q016': 'category',
        'Q017': 'category',
        'Q018': 'category',
        'Q019': 'category',
        'Q020': 'category',
        'Q021': 'category',
        'Q022': 'category',
        'Q023': 'category',
        'Q024': 'category',
        'Q025': 'category',
        'TP_ESCOLA': 'category',
        'TP_ENSINO': 'category',
        'TP_FAIXA_ETARIA': 'category'
    }

    logger.info('Loading data...')
    s3 = boto3.client('s3')
    bucket_name = 'tc3grupo46'
    file_key = 'refined/part-00000-b3cc79a7-0d3a-4881-bc50-b2b38a5da27a-c000.csv'  # noqa

    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv = response['Body'].read().decode('utf-8')

    csv_file = io.StringIO(csv)

    df = pd.read_csv(csv_file, sep=',', header=0, dtype=dtype_dict)

    columns_for_training = [
        'Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006',
        'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012',
        'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018',
        'Q019', 'Q020', 'Q021', 'Q022', 'Q023', 'Q024',
        'Q025', 'TP_ESCOLA', 'TP_ENSINO', 'TP_FAIXA_ETARIA'
    ]

    X_train = df.loc[:, columns_for_training]

    logger.info('Data loaded from S3')

    # predict columns
    y_nota_ch = df['NU_NOTA_CH']
    y_nota_cn = df['NU_NOTA_CN']
    y_nota_lc = df['NU_NOTA_LC']
    y_nota_mt = df['NU_NOTA_MT']
    y_nota_redacao = df['NU_NOTA_REDACAO']

    logger.info('Training models...')

    logger.info('Training model nota_ch...')
    start_time = datetime.now()
    train_model_XGBoost(X_train, y_nota_ch, 'nota_ch')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_ch trained')

    logger.info('Training model nota_cn...')
    start_time = datetime.now()
    train_model_XGBoost(X_train, y_nota_cn, 'nota_cn')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_cn trained')

    logger.info('Training model nota_lc...')
    start_time = datetime.now()
    train_model_XGBoost(X_train, y_nota_lc, 'nota_lc')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_lc trained')

    logger.info('Training model nota_mt...')
    start_time = datetime.now()
    train_model_XGBoost(X_train, y_nota_mt, 'nota_mt')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_mt trained')

    logger.info('Training model nota_redacao...')
    start_time = datetime.now()
    train_model_XGBoost(X_train, y_nota_redacao, 'nota_redacao')
    end_time = datetime.now()
    execution_time = end_time - start_time
    formatted_time = str(execution_time).split('.')[0]
    logger.info(f"Execution time: {formatted_time}")
    logger.info('Model nota_mt nota_redacao')

    logger.info('Models trained!')
