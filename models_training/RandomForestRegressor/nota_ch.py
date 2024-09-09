from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(max_depth=16)
    model.fit(X_train, y_train)
    with open('../models_pickle/model_nota_ch.pkl', 'wb') as f:
        pickle.dump(model, f)
