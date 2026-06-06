import pandas as pd
import lightgbm as lgb
from preprocessing import engineer_all_features
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

def model_lgbm(df):

    X, y, prices, dates = engineer_all_features(df)
    date_now = pd.Timestamp.now()

    training_mask = dates < date_now
    X_train = X[training_mask]
    y_train = y[training_mask]

    validation_date = date_now - pd.DateOffset(months = 2)
    validation_mask = (dates >= validation_date) & (dates < date_now)
    X_val = X[validation_mask]
    y_val = y[validation_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_data = lgb.Dataset(X_train_scaled, label = y_train)
    val_data = lgb.Dataset(X_val_scaled, label = y_val, reference = train_data)

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'verbose': -1,
        'learning_rate': 0.09,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds = 200), 
        lgb.log_evaluation(period = 100)
        ]
    
    model = lgb.train(params, train_data, num_boost_round = 2000, valid_sets = [val_data], callbacks = callbacks)

    y_val_pred = model.predict(X_val_scaled)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = root_mean_squared_error(y_val, y_val_pred)

    print(f"Validation R2: {val_r2:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    
    return model, scaler, params

