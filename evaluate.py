import joblib
import sys
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_test_data_for_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae

if __name__ == "__main__":
    
    model = joblib.load('model.pkl')
    
    if len(sys.argv) > 2:
        X_path = sys.argv[1]
        y_path = sys.argv[2]
    else:
        X_path = 'X_sample.csv'
        y_path = 'y_sample.csv'
    
    with mlflow.start_run(experiment_id=0):
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)
    
        y_pred = model.predict(X)
        
        r2, mae = evaluate_test_data_for_model(y, y_pred)
        
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mae', mae)
    