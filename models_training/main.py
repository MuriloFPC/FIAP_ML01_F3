import pandas as pd
from RandomForestRegressor.nota_ch import train_model as train_model_ch
from datetime import datetime

# Load data
print('Loading data...')
csv = pd.read_csv('../csv/Completo.csv', sep=';', encoding='latin-1')
print('Data loaded')
print('Preprocessing data...')
csv.dropna(inplace=True)
print('Data preprocessed')

colunas_remover = [x for x in csv.columns[0:51] if x not in ['TP_ESCOLA']]

X = csv.drop(columns=colunas_remover)
y_nota_ch = csv['NU_NOTA_CH'] #predict column
y_nota_cn = csv['NU_NOTA_CN'] #predict column
y_nota_lc = csv['NU_NOTA_LC'] #predict column
y_nota_mt = csv['NU_NOTA_MT'] #predict column
y_nota_redacao = csv['NU_NOTA_REDACAO'] #predict column


mymap = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6,'G':7,'H':8, 'I':9, 'J':10, 'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P': 16, 'Q':17}

print('Ajustandando arquivo')
X = X.map(lambda s: mymap.get(s) if s in mymap else s)
print('Arquivo ajustado')

print('Training models...')
# Train models
print('Training model nota_ch...')
start_time = datetime.now()
train_model_ch(X, y_nota_ch)
end_time = datetime.now()
execution_time = end_time - start_time
formatted_time = str(execution_time).split('.')[0]
print(f"Execution time: {formatted_time}")
print('Model nota_ch trained')