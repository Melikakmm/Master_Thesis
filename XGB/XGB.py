
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.metrics import mean_squared_error
import io
import xgboost as xgb
from xgboost import XGBRegressor
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import pandas as pd
import xgboost as xgb
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pickle


# This is a dataset I found in Dekany's model, he trained his model on this dataset.

file = './data/o4rrab_gaiaDR2_bp_rp_i_g_param.dat'



data = []
header = []
with open(file, 'r') as file:
    for line in file:
        out_line=line
        if line.startswith('# '):
            out_line=line.replace('# ', '')
            header.append(out_line)
        else:
            line = line.split()
            data.append(line)
        

df = pd.DataFrame()

df = pd.DataFrame([[x for x in row] for row in data])
df[0] = df[0].astype(int)
header[0].split()
head_dict = {key: value for key, value in zip(header[0].split(), df.columns) }
df.columns = head_dict
df.drop_duplicates(inplace= True)
df.reset_index(drop = True)


#these are for training the models:
#x = df[['period', 'A1_g', 'A2_g', 'A3_g', 'phi31_g', 'phi21_g']]
x = df[['period', 'A1_g', 'A2_g', 'A3_g', 'phi31_g', 'phi21_g']]
y = df[['FeH']]














features = [['period', 'phi31_g', 'phi21_g'], ['period', 'A1_g', 'A2_g'],
            ['period', 'A1_g', 'phi31_g'], ['period', 'phi31_g'],
            ['period', 'A1_g', 'A2_g', 'A3_g', 'phi31_g', 'phi21_g']]


def tunner(input_, features = features, test_size = [0.1,0.2, 0.25, 0.3], split = [5, 10, 15]):
    best_model_list = []
    params_total = []
    for f in features:
        x = input_[f]
        for size in test_size :
            for n_split in split:
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=size, shuffle = True, random_state=42)

                X_train = np.array(X_train, dtype = float)
                y_train = np.array(y_train, dtype = float)
                X_test = np.array(X_test, dtype = float)
                y_test = np.array(y_test, dtype = float)


                model = xgb.XGBRegressor(objective='reg:squarederror')

                param_grid = {'colsample_bytree': [0.8, 0.9, 1.0],
                 'learning_rate': [0.05, 0.06, 0.04, 0.005],
                 'max_depth': [5,7, 8, 10],
                 'n_estimators': [100, 200, 300],
                 'subsample': [0.8, 0.9, 1.0], 'objective': ['reg:squarederror']}
                
                
                
                kf = KFold(n_splits=n_split, shuffle=True, random_state=42)
                
                
                grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                
                best_params = grid_search.best_params_
                best_model = grid_search.best_estimator_
                model = xgb.XGBRegressor(**best_params)
                model_XG = model.fit(X_train,y_train)
                y_test_pre = model_XG.predict(X_test)
                score = mean_squared_error(y_test, y_test_pre)
                dict_ = {'parameters': best_params, 'test_size': size, 'split': n_split, 'features': f, 'mse': score}
                params_total.append(dict_)
                
                
tunner(x) 



with open("total_params.pkl", "wb") as pickle_file:
    pickle.dump(data, pickle_file)
