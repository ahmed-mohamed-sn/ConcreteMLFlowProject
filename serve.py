import joblib
import sys
import pandas as pd

if __name__ == "__main__":
    
    model = joblib.load('model.pkl')
    
    file_path = sys.argv[1]
    
    sample_df = pd.read_csv(file_path)
    
    results = model.predict(sample_df)
    
    print(results)
    