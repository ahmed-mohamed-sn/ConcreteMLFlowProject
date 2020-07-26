import joblib
import sys
import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_test_data_for_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print('R2: {}'.format(r2))
    print('MAE: {}'.format(mae))
    return r2, mae

if __name__ == "__main__":
    
    model = joblib.load('model.pkl')
    
    file_path = sys.argv[1]
    
    with mlflow.start_run(experiment_id=0):
        sample_df = pd.read_csv(file_path)
    
        y_pred = model.predict(sample_df)
        
        r2, mae = evaluate_test_data_for_model(rf_model, X_val, y_val)
        
        mlflow.log_metric('r2', r2)
        mlflow.log_metric('mae', mae)
    