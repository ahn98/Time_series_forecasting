# Optuna Libraries
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('time', inplace=True)
    return df

def split_data(df, target_col, test_size=0.4, val_size=0.5, random_state=2021):
    feature_columns = list(df.columns.difference([target_col]))
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=val_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_val, y_val

def tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10):
    # optuna
    # random sampler
    sampler = TPESampler(seed=seed)

    # define function
    def objective(trial):
        xgb_param = {
        'tree_method' : 'gpu_hist',
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'max_leaves': trial.suggest_int('max_leaves', 5, 300, step=1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
        }

        # Generate model
        model_xgb = xgb.XGBRegressor(**xgb_param)
        model_xgb = model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                           verbose=0, early_stopping_rounds=50)
                           
        # 평가 지표
                        
        MSE = mean_squared_error(y_val, model_xgb.predict(X_val))
        return MSE

    optuna_xgboost = optuna.create_study(direction='minimize', sampler=sampler)
    optuna_xgboost.optimize(objective, n_trials=n_trials)
    xgboost_trial = optuna_xgboost.best_trial
    xgboost_trial_params = xgboost_trial.params
    print('Best Trial: score {},\nparams {}'.format(xgboost_trial.value, xgboost_trial_params))
    return xgboost_trial_params

def train_model(X_train, y_train, hyperparameters):
    xgboost = XGBRegressor(**hyperparameters)
    xgboost.fit(X_train, y_train)
    return xgboost

def fit_xgb(X_train, y_train, X_test, y_test, xgboost_trial_params):
    # LGBM Regressor fit
    xgboost = XGBRegressor(**xgboost_trial_params)
    xgboost = xgboost.fit(X_train, y_train)

    # Predict the y_test
    pred = xgboost.predict(X_test)

    # results
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print("RMSE: %f" % (rmse))

    mae = mean_absolute_error(y_test, pred)
    print('mae: %f' %(mae))  

    r2 = r2_score(y_test, pred)
    print('R2: %f' %(r2))

########################
######## Model #########
########################

target_col = "Tair"   # 타겟 설정
X_train, X_test, y_train, y_test, X_val, y_val = split_data(gcwactu3, target_col)
scaler11 = MinMaxScaler()
scaler22 = MinMaxScaler()
X_train = scaler11.fit_transform(X_train)
X_test = scaler11.fit_transform(X_test)
X_val = scaler11.fit_transform(X_val)

y_train_array = np.array(y_train)
y_train_reshape = y_train_array.reshape(-1, 1)
y_train = scaler22.fit_transform(y_train_reshape)

y_test_array = np.array(y_test)
y_test_reshape = y_test_array.reshape(-1, 1)
y_test = scaler22.fit_transform(y_test_reshape)

y_val_array = np.array(y_val)
y_val_reshape = y_val_array.reshape(-1, 1)
y_val =scaler22.fit_transform(y_val_reshape)

best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials=100, seed=10)
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

model = train_model(X_train, y_train, best_params)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print('Tair MSE: {:.4f}'.format(mse)) 
print('Tair MAE: {:.4f}'.format(mae))   
print('Tair RMSE: {:.4f}'.format(rmse))  
print('Tair R^2: {:.4f}'.format(r2))
