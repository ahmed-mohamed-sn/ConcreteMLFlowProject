import joblib
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.datasets import load_concrete

def evaluate_test_data_for_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'R2: {r2}')
    print(f'MAE: {mae}')
    return r2, mae

if __name__ == "__main__":
    
    min_samples_split = float(sys.argv[1]) if len(sys.argv) > 1 else 2
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_estimators = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    max_features = sys.argv[4] if len(sys.argv) > 4 else 'auto'
    
    dataset = load_concrete(return_dataset=True)
    df = dataset.to_dataframe()
    
    target = 'strength'
    X = df.drop(target, axis=1).copy()
    numerical_features = X.columns.tolist()
    Y = df.loc[:, target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=77)
    
    with mlflow.start_run(experiment_id=0):
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_features', max_features)
        
        rf_model = RandomForestRegressor(min_samples_split=min_samples_split,
                                         max_depth=max_depth,
                                         n_estimators=n_estimators,
                                         max_features=max_features,
                                         random_state=7)
        rf_model.fit(X_train, y_train)
    
        r2, mae = evaluate_test_data_for_model(rf_model, X_test, y_test)
        
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mae', mae)
        
        mlflow.sklearn.log_model(rf_model, artifact_path='model')
    